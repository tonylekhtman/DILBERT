# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from collections import Counter, defaultdict

from nltk import sent_tokenize
from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import f1_score as reg_f1_score
from seqeval.metrics.sequence_labeling import get_entities
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer

from cat_config import get_cats
from max_sim import get_max_cat_similarity, calc_similarity
# from sentence_embeddings import get_best_examples
from other_eval import evaluate_chunk
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
# #     (tuple(conf.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP .keys()) for conf in (BertConfig, RobertaConfig)),
# #     ())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=f'runs/{os.path.basename(args.output_dir)}')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps * t_total,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        non_trained_steps = random.sample(range(len(epoch_iterator)), int(args.ner_dropout * len(epoch_iterator)))
        for step, batch in enumerate(epoch_iterator):

            if step in non_trained_steps:
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def my_f1_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len([x for x in pred_entities if
                      any(len(set(range(t[1], t[2] + 1)) & set(range(x[1], x[2] + 1))) > 0 for t in true_entities)])
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, file_path=args.eval_file_path,
                                           mode=mode)
    if not eval_dataset:
        return None, None
    label_map = {i: label for i, label in enumerate(labels)}

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    samples = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            samples = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            samples = np.append(samples, inputs["input_ids"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    prob_preds = torch.sigmoid(torch.Tensor(preds))

    preds = np.argmax(preds, axis=2)
    # preds_final = []
    # for p in preds:
    #    preds_final.append([np.argmax(x) if x[2] > args.f1_threshold else np.argmax(x[0:2]) for x in np.array(torch.nn.Softmax()(torch.Tensor(p)))])

    # preds = preds_final
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    samples_list = [[] for _ in range(out_label_ids.shape[0])]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
                samples_list[i].append(tokenizer.ids_to_tokens[samples[i][j]])
            else:
                if len(samples_list[i]) != 0 and samples[i][j] not in tokenizer.all_special_ids:
                    samples_list[i][-1] = samples_list[i][-1] + tokenizer.ids_to_tokens[samples[i][j]].strip('##')

    results = {
        f"{prefix}loss": eval_loss,
        # f"{prefix}precision": precision_score(out_label_list, preds_list),

        # f"{prefix}recall": recall_score(out_label_list, preds_list),
        # f"{prefix}f1": f1_score(out_label_list, preds_list),
        f"{prefix}precision": evaluate_chunk(out_label_list, preds_list)[0],
        f"{prefix}recall": evaluate_chunk(out_label_list, preds_list)[1],
        f"{prefix}f1": evaluate_chunk(out_label_list, preds_list)[2],
        f"{prefix}f1_only_gt_with_as>0": f1_score(
            np.array(out_label_list)[[y for y, x in enumerate(out_label_list) if any(i != 'O' for i in x)]],
            np.array(preds_list)[[y for y, x in enumerate(out_label_list) if any(i != 'O' for i in x)]]),
        f"{prefix}f1_recognizing_no_aspect": reg_f1_score(
            [1 if len(get_entities(x)) == 0 else 0 for x in out_label_list],
            [1 if len(get_entities(x)) == 0 else 0 for x in preds_list]),
        f"{prefix}f1_recognizing_more_than_1_aspect": reg_f1_score(
            [1 if len(get_entities(x)) > 0 else 0 for x in out_label_list],
            [1 if len(get_entities(x)) > 0 else 0 for x in preds_list]),
        f"{prefix}partial_f1": my_f1_score(out_label_list, preds_list)
    }

    false_positives_accumulated = []
    false_positive_samples = defaultdict(list)
    false_negative_samples = defaultdict(list)
    false_negatives_accumulated = []
    sampled_ids = range(len(samples_list))  # random.sample(range(len(samples_list)), 30)

    verbose_evaluation = False
    if verbose_evaluation:
        from gensim.models import FastText
        embedding_model = FastText.load_fasttext_format('embeddings/cc.en.300.bin')

        domain_categories = get_cats(task.split('_')[1])
        with open(
                f'sample_examples/{os.path.basename(args.model_name_or_path)}_{os.path.dirname(args.data_dir)}_{os.path.basename(args.data_dir)}',
                'w+') as output_sample_example:
            for i, (sample, prediction, gt) in enumerate(zip(samples_list, preds_list, out_label_list)):
                if i in sampled_ids:
                    predicted_entities_indexes = get_entities(prediction)
                    true_entities_indexes = get_entities(gt)
                    common_indexes = (set(predicted_entities_indexes) & set(true_entities_indexes))
                    false_positive_entities_indexes = set(predicted_entities_indexes) - common_indexes
                    false_negative_entities_indexes = set(true_entities_indexes) - common_indexes
                    predicted_entities = get_entities_text(predicted_entities_indexes, sample)
                    true_entities = get_entities_text(true_entities_indexes, sample)
                    false_positives = get_entities_text(false_positive_entities_indexes, sample)
                    false_negatives = get_entities_text(false_negative_entities_indexes, sample)
                    if len(false_negatives) == 0 and len(false_positives) == 0 and len(true_entities) > 0:
                        with open(
                                f'sample_examples_corrects/{os.path.basename(args.model_name_or_path)}_{os.path.dirname(args.data_dir)}_{os.path.basename(args.data_dir)}',
                                'a+') as output_corrects:
                            output_corrects.write(
                                f'{" ".join(sample)}###{",".join([" ".join(t) for t in true_entities])}\n')

                    print(f'sample: {" ".join(sample)}')
                    output_sample_example.write(f'sample: {" ".join(sample)}' + '\n')
                    print('masked sample:')
                    similarities = []
                    for word in sample:
                        similarity = get_max_cat_similarity(word, domain_categories, embedding_model, calc_similarity)
                        similarities.append(similarity)
                    sim = np.percentile(similarities, 90)
                    print(' '.join(
                        [f'{word}' if word_sim < sim else f'##{word}## ({word_sim})' for word, word_sim in
                         zip(sample, similarities)]))
                    output_sample_example.write('masked example:' + ' '.join(
                        [f'{word}' if word_sim < sim else f'##{word}## ({word_sim})' for word, word_sim in
                         zip(sample, similarities)]) + '\n')
                    print('\n')
                    print('predicted entities:')
                    output_sample_example.write('predicted entities:' + '\n')
                    for predicted_entity in predicted_entities:
                        print(f"\t{' '.join(predicted_entity)}")
                        output_sample_example.write(' '.join(predicted_entity) + '\n')
                        # print('\n')
                    print('\n')
                    print('true entities:')
                    output_sample_example.write('true_entities:' + '\n')
                    for true_entity in true_entities:
                        print(f"\t{' '.join(true_entity)}")
                        output_sample_example.write(' '.join(true_entity) + '\n')
                        # print('\n')
                    print('\n')
                    print('false positives:')
                    output_sample_example.write('false positives:' + '\n')
                    for fp_entity in false_positives:
                        fp_entity_text = ' '.join(fp_entity)
                        false_positives_accumulated.append(fp_entity_text)
                        false_positive_samples[fp_entity_text].append(sample)
                        print(f'\t{fp_entity_text}')
                        output_sample_example.write(fp_entity_text + '\n')
                    print('\n')
                    print('false negatives:')
                    output_sample_example.write('false negatives:' + '\n')
                    for fn_entity in false_negatives:
                        fn_entity_text = ' '.join(fn_entity)
                        false_negatives_accumulated.append(fn_entity_text)
                        false_negative_samples[fn_entity_text].append(sample)
                        print(f'\t{fn_entity_text}')
                        output_sample_example.write(fn_entity_text + '\n')

                # for word, word_sim in zip(sample, similarities):
                #     if word_sim > sim:
                #         print(f'word:{word}')
                #         print(f'word_sim: {word_sim}')

            print(f'Summary of false positives:')
            print(Counter(false_positives_accumulated))
            print(f'Summary of false negatives:')
            print(Counter(false_negatives_accumulated))
            print(false_positive_samples['price'])
            print(false_negative_samples['food'])
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def get_entities_text(predicted_entities_indexes, sample):
    predicted_entities = []
    for predicted_entity_index in predicted_entities_indexes:
        predicted_entity_start = predicted_entity_index[1]
        predicted_entity_end = predicted_entity_index[2]
        entity_text = sample[predicted_entity_start:predicted_entity_end + 1]
        predicted_entities.append(entity_text)
    return predicted_entities


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, file_path, mode):
    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", file_path)

    examples = read_examples_from_file(file_path, mode)
    if len(examples) == 0:
        return
    features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(args.model_type in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(args.model_type in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset




def run_ner_with_args(args):
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)
    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model2 = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    # config.output_hidden_states = True
    model = BertForTokenClassification(config=config)
    model.bert.load_state_dict(model2.bert.state_dict())

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id,file_path=args.train_file_path, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result1, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
            if global_step:
                result1 = {"{}_{}".format(global_step, k): v for k, v in result1.items()}
            results.update(result1)

        output_eval_file = f"eval_results.txt"
        with open(output_eval_file, "a+") as writer:
            writer.write(f'{args.model_name_or_path}\n')
            writer.write(f"{os.path.basename(args.train_file_path)}\n")
            writer.write(f"{os.path.basename(args.eval_file_path)}\n")
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "dev.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
    return results


