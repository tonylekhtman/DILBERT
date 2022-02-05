import hashlib
import itertools
import os
import json
import time
from shutil import copyfile

import logging

import socket

from benchmark import run_benchmark
from classification_data_creation import create_dataset, create_classification_dataset_for_threshold, \
    create_classification_sum_tokens, create_classification_num_of_top_categories
from max_sim import max_sim_parser, create_index
from run_class import run_classification_task
from run_lm_finetuning import main as run_lm, parser
import argparse
import sys

base_parser = argparse.ArgumentParser()
base_parser.add_argument('--gpu', default='0', )
base_parser.add_argument('--training_file', '-t', default='trainings/sample.yaml')
args = base_parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpu_num = args.gpu
import yaml


def create_custom_embeddings(unlabeled_data_files, output_path):
    if not os.path.exists('fastText-0.9.2/fasttext'):
        os.system('wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip')
        os.system('rm -rf fastText-0.9.2')
        os.system('unzip v0.9.2.zip')
        os.system('cd fastText-0.9.2;make')
        os.system('cd ..')
    os.system(f'cat {" ".join(unlabeled_data_files)} > tmp_file.txt')
    fasttext_cmd = f'fastText-0.9.2/fasttext skipgram -input tmp_file.txt -output {output_path}'
    os.system(fasttext_cmd)


def add_unigram_masking_params(list_of_args):
    content_hashes = {domain: hashlib.md5((open(f'unlabeled_data/{domain}.raw', 'rb').read())).hexdigest() for domain in
                      selected_domain_names}
    for domain in selected_domain_names:
        if not os.path.exists(f'similarity_cache/{domain}_{content_hashes[domain]}.json'):
            embedding_model = d.get('embedding_model', 'fasttext')
            max_sim_args = ['--embedding_model', embedding_model, '--domain', domain, '--n', '1', '--output',
                            f'similarity_cache/{domain}_{content_hashes[domain]}.json',
                            '--path', f'unlabeled_data/{domain}.raw']
            if embedding_model == 'custom_fasttext':
                hashes = [content_hashes[f] for f in selected_domain_names]
                embedding_model_path = f"embeddings/{'_'.join(hashes)}"

                max_sim_args += ['--embedding_model_path', embedding_model_path]
                if not os.path.exists(f"{embedding_model_path}.bin"):
                    create_custom_embeddings(
                        [f'unlabeled_data/{x}.raw' for x in selected_domain_names],
                        embedding_model_path)
            create_index(max_sim_parser.parse_args(max_sim_args))
    unigram_indexes = [f'similarity_cache/{domain}_{content_hashes[domain]}.json' for domain in selected_domain_names]
    thresholds = [d['mlm_thresholds'][domain] for domain in selected_domain_names]
    list_of_args += ['--custom_masking', '--unigram_indexes'] + unigram_indexes
    list_of_args += ['--thresholds'] + [str(t) for t in thresholds]
    return list_of_args


training_file = args.training_file
try:
    d = yaml.load(open(training_file))
    selected_domain_names = d['selected_domains']
    content_hashes = {domain: hashlib.md5((open(f'unlabeled_data/{domain}.raw', 'rb').read())).hexdigest() for domain in
                      selected_domain_names}
    if d.get('embedding_model', 'fasttext') == 'custom_fasttext':
        hashes = [content_hashes[f] for f in selected_domain_names]
        embedding_model_path = f"embeddings/{'_'.join(hashes)}"
    else:
        embedding_model_path = f"embeddings/cc.en.300"
    with open(f'{training_file}', 'w') as file:
        d['gpu_num'] = gpu_num
        documents = yaml.dump(d, file)

    list_of_args = ["--mlm", "--do_train", "--do_lower_case", "--mlm_probability", str(d.get("base_mlm_prob", 0.15))]
    if not d.get('gpu', True):
        list_of_args += ['--no_cuda']
    if d.get('mlm_randomness', True):
        list_of_args += ['--mlm_randomness']
    hashed_content = '_'.join(content_hashes[x] for x in selected_domain_names)
    if 'mlm_thresholds' in d:
        selected_thresholds = '_'.join([str(d['mlm_thresholds'][x]) for x in selected_domain_names])
    else:
        selected_thresholds = ''
    output_path = d.get('output_path', f'{d["pre_trained_model_name_or_path"]}_{hashed_content}_{selected_thresholds}_{d.get("embedding_model", "fasttext")}')
    list_of_args += ["--output_dir", output_path]
    model_type = d['model_type']
    model_name_or_path = d['pre_trained_model_name_or_path']
    seed = d['seed']
    num_train_epochs = str(d.get('num_cmlm_epochs', 1.0))
    list_of_args += ['--model_type', model_type, '--model_name_or_path', model_name_or_path, '--seed',
                     str(seed),
                     '--save_steps', '0', '--num_train_epochs', num_train_epochs]
    list_of_args += ['--domains'] + selected_domain_names
    file_paths = [f'unlabeled_data/{domain}.raw' for domain in selected_domain_names]
    list_of_args += ['--file_paths'] + file_paths

    if d['masking'] == 'unigram':
        list_of_args = add_unigram_masking_params(list_of_args)

    args = parser.parse_args(list_of_args)
    print(args)
    if not os.path.exists(output_path):
        run_lm(args)
    output_models = [output_path]
    if d['classification']:
        reviews_to_similarities = {}
        for domain_name, file_path in zip(selected_domain_names, file_paths):
            reviews_to_similarities[domain_name] = create_dataset(
                file_path,
                domain_name,
                f'reviews_cache_{domain_name}_{os.path.basename(embedding_model_path)}.json',
                embedding_model_path, d)

        if d.get('classification_type', 'threshold') == 'threshold':
            if d.get('classification_domain'):
                selected_domain_names_for_threshold = [d['classification_domain']]
                classification_thresholds = [float(d['classification_thresholds'][d['classification_domain']])]
            else:
                selected_domain_names_for_threshold = selected_domain_names
                classification_thresholds = [float(d['classification_thresholds'][domain]) for domain in
                                             selected_domain_names]
            classification_dataset_path = create_classification_dataset_for_threshold(reviews_to_similarities,
                                                                                      selected_domain_names_for_threshold,
                                                                                      classification_thresholds,
                                                                                      d[
                                                                                          'classification_samples_per_domain'],
                                                                                      d['seed'])

        output_class_model_path = os.path.splitext(os.path.basename(classification_dataset_path))[
                                      0] + f'_pt_model_{os.path.basename(d["output_path"])}'
        output_dir = f'{output_class_model_path}'
        if not os.path.exists(output_dir):
            classification_acuuracy, classification_loss = run_classification_task(classification_dataset_path,
                                                                                   d['classification_epochs'],
                                                                                   d['output_path'], output_dir)
            d['classification_accuracy'] = classification_acuuracy
            d['classification_loss'] = classification_loss
        class_models = []
        for i in range(d['classification_epochs']):
            class_models.append(f'{output_dir}_{i}')
        output_models = class_models
    tasks = d['tasks']
    d['server'] = socket.gethostname()
    d['training_file'] = training_file
    d['num_ner_epochs'] = d.get('num_ae_epochs', 3)
    d['ner_dropout'] = d.get('ner_dropout', 0)
    d['f1_threshold'] = d.get('f1_threshold', 0.5)
    d['ae_lr'] = d.get('ae_lr', 5e-5)
    d["train_batch_size"] = d.get('ae_train_batch_size', 32)
    run_benchmark(output_models, tasks, d['absa_dirs'], os.path.basename(training_file),
                  d.get('gpu', True), d)
except Exception as e:
    logging.exception("error")
