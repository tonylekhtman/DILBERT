import argparse
import os
import tempfile
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import uuid
from nltk import sent_tokenize

from run_ae import MODEL_CLASSES, run_ae_with_args

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--train_file_path", default=None, type=str, required=True,
                    help="The train file path in CoNLL-2003 format")
parser.add_argument("--eval_file_path", default=None, type=str, required=True,
                    help="The eval file path in CoNLL-2003 format")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name selected in the list: ")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--labels", default="", type=str,
                    help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--do_train", action="store_true",
                    help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true",
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_predict", action="store_true",
                    help="Whether to run predictions on the test set.")
parser.add_argument("--evaluate_during_training", action="store_true",
                    help="Whether to run evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action="store_true",
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=50,
                    help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=0,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action="store_true",
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", action="store_true",
                    help="Avoid using CUDA when available")
parser.add_argument("--overwrite_output_dir", action="store_true",
                    help="Overwrite the content of the output directory")
parser.add_argument("--overwrite_cache", action="store_true",
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--fp16", action="store_true",
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
parser.add_argument('--ae_dropout', type=float, default=0)
parser.add_argument('--ae_sampled_examples', action='store_true')
parser.add_argument('--tgt_snts_file', type=str)
parser.add_argument('--sampling_model_path', type=str)
parser.add_argument('--ae_sampled_percent', type=float, default=0.8)
parser.add_argument('--f1_threshold', type=float)


def generate_output_model_name():
    return next(tempfile._get_candidate_names())


def run_single_task(train_file_path, eval_file_path, task, pt_model, data_dirs, gpu=True, config=None):
    if config is not None:
        config['task'] = task
    output = generate_output_model_name()
    metrics = defaultdict(list)
    for data_dir in data_dirs:
        seed = config['seed']  # random.randrange(100000)

        ae_args = ["--train_file_path", train_file_path, "--eval_file_path", eval_file_path, "--model_type", "bert",
                    "--model_name_or_path",
                    f"{pt_model}", "--output",
                    f"ae_models/{output}", "--labels", "ae_files/labels.txt", "--do_eval",
                    "--seed", str(seed), "--save_steps", "750", "--per_gpu_train_batch_size",
                    str(config['train_batch_size']),
                    "--max_seq_length", "128", "--overwrite_output_dir", "--do_lower_case", "--do_train",
                    "--num_train_epochs", str(config['num_ae_epochs']), '--ae_dropout', str(config['ae_dropout']),
                    '--ae_sampled_examples', '--tgt_snts_file', 'stam', '--sampling_model_path', f"{pt_model}",
                    '--f1_threshold', str(config['f1_threshold']), '--learning_rate', str(config['ae_lr'])]
        if not gpu:
            ae_args += ['--no_cuda']
        args = parser.parse_args(
            ae_args)
        results = run_ae_with_args(args)
        for metric in results:
            metrics[metric].append(results[metric])
    mean_results = {}
    for k in metrics:
        mean_results[k] = np.mean(metrics[k])
        mean_results[f'{k}_std'] = np.std(metrics[k])
    return mean_results


def run_benchmark(models, tasks, ae_data_files_dirs, unique_id=None, gpu=True, config=None):
    dfs = []
    num_seeds = 1
    for model in models:
        final_results = {}
        for task in tasks:
            splitted_task = task.split('_')
            train_file_path = f'ae_files/{splitted_task[0]}/train.txt'
            eval_file_path = f'ae_files/{splitted_task[-1]}/dev.txt'
            f1_res = run_single_task(train_file_path, eval_file_path, task, model, ae_data_files_dirs, gpu,
                                     config)
            model_name = os.path.basename(model)
            final_results[f'{model_name}_{task}'] = f1_res
            # final_results[f'{model_name}']['task'] = task
        df = pd.DataFrame(final_results)
        dfs.append(df)
    res = pd.concat(dfs, axis=1)
