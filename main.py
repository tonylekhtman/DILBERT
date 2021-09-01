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
base_parser.add_argument('--gpu',default='0', )
base_parser.add_argument('--training_file','-t', default='trainings/sample2.yaml')
args = base_parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpu_num = args.gpu
import yaml


def create_custom_embeddings(unlabeled_data_files, output_path):
    os.system(f'cat {" ".join(unlabeled_data_files)} > tmp_file.txt')
    fasttext_cmd = f'./embeddings/fastText-0.9.2/fasttext skipgram -input tmp_file.txt -output {output_path}'
    os.system(fasttext_cmd)


def add_unigram_masking_params(list_of_args):
    for domain in [x for x in d['domains'] if x['name'] in d['selected_domains']]:
        if not os.path.exists(domain['unigram_index']):
            embedding_model = d.get('embedding_model', 'fasttext')
            max_sim_args = ['--embedding_model', embedding_model, '--domain', domain['name'], '--n', '1', '--output',
                            domain['unigram_index'],
                            '--path', domain['file_path']]
            if embedding_model == 'custom_fasttext':
                max_sim_args += ['--embedding_model_path', d['embedding_model_path']]
                if  not os.path.exists(f"{d['embedding_model_path']}.bin"):
                    create_custom_embeddings([x['file_path'] for x in d['domains'] if x['name'] in d['selected_domains']], d['embedding_model_path'])
            create_index(max_sim_parser.parse_args(max_sim_args))
    unigram_indexes = [domain['unigram_index'] for domain in d['domains'] if
                       domain['name'] in d['selected_domains']]
    thresholds = [str(domain['threshold']) for domain in d['domains'] if
                  domain['name'] in d['selected_domains']]
    list_of_args += ['--custom_masking', '--unigram_indexes'] + unigram_indexes
    list_of_args += ['--thresholds'] + thresholds
    return list_of_args



training_file = args.training_file
try:
    d = yaml.load(open(training_file))
    with open(f'{training_file}', 'w') as file:
        d['gpu_num'] = gpu_num
        documents = yaml.dump(d, file)

    list_of_args = ["--mlm", "--do_train", "--do_lower_case", "--mlm_probability", str(d.get("base_mlm_prob",0.15))]
    if not d.get('gpu', True):
        list_of_args += ['--no_cuda']
    if d.get('mlm_randomness', True):
        list_of_args += ['--mlm_randomness']
    output_path = d['output_path']
    list_of_args += ["--output_dir", output_path]
    model_type = d['model_type']
    model_name_or_path = d['model_name_or_path']
    seed = d['seed']
    save_steps = d['save_steps']
    num_train_epochs = str(d.get('num_train_epochs', 1.0))
    list_of_args += ['--model_type', model_type, '--model_name_or_path', model_name_or_path, '--seed',
                     str(seed),
                     '--save_steps', str(d['save_steps']), '--num_train_epochs', num_train_epochs]
    domain_names = [domain['name'] for domain in d['domains'] if domain['name'] in d['selected_domains']]
    list_of_args += ['--domains'] + domain_names
    file_paths = [domain['file_path'] for domain in d['domains'] if domain['name'] in d['selected_domains']]
    list_of_args += ['--file_paths'] + file_paths

    if d['masking'] == 'unigram':
        list_of_args = add_unigram_masking_params(list_of_args)
    elif d['masking'] == 'bigram':
        list_of_args = add_unigram_masking_params(list_of_args)
        for domain in [x for x in d['domains'] if x['name'] in d['selected_domains']]:
            if not os.path.exists(domain['bigram_index']):
                max_sim_args = ['--fasttext', '--domain', domain['name'], '--n', '2', '--output',
                                domain['bigram_index'],
                                '--path', domain['file_path']]
                create_index(max_sim_parser.parse_args(max_sim_args))

        bigram_indexes = [domain['bigram_index'] for domain in d['domains'] if
                          domain['name'] in d['selected_domains']]

        list_of_args += ['--bigram_masking', '--bigram_indexes'] + bigram_indexes
        bigram_thresholds = [str(domain.get('bigram_threshold', domain['threshold'])) for domain in d['domains']
                             if
                             domain['name'] in d['selected_domains']]
        list_of_args += ['--bigram_thresholds'] + bigram_thresholds

    args = parser.parse_args(list_of_args)
    print(args)
    if not os.path.exists(output_path):
        run_lm(args)
    output_models = [output_path]
    if d['classification']:
        reviews_to_similarities = {}
        for domain_name, file_path in zip(domain_names, file_paths):
            reviews_to_similarities[domain_name] = create_dataset(
                file_path,
                domain_name,
                f'reviews_res_{domain_name}_sentence_{os.path.basename(d.get("embedding_model_path",""))}.json', d)

        if d['classification_type'] == 'threshold':
            classification_thresholds = [float(domain['class_threshold']) for domain in d['domains'] if
                                         domain['name'] in d['selected_domains']]
            if d.get('classification_domain'):
                selected_domain_names_for_threshold = [d['classification_domain']]
                classification_thresholds = [float(domain['class_threshold']) for domain in d['domains'] if
                                             domain['name'] == d['classification_domain']]
            else:
                selected_domain_names_for_threshold = domain_names
                classification_thresholds = [float(domain['class_threshold']) for domain in d['domains'] if
                                             domain['name'] in d['selected_domains']]
            classification_dataset_path = create_classification_dataset_for_threshold(reviews_to_similarities,
                                                                                      selected_domain_names_for_threshold,
                                                                                      classification_thresholds,
                                                                                      d[
                                                                                          'classification_samples_per_domain'],
                                                                                      d['seed'])

        elif d['classification_type'] == 'sum_tokens':
            if d.get('classification_domain'):
                selected_domain_names_for_threshold = [d['classification_domain']]
                classification_alphas = [float(domain['class_alpha']) for domain in d['domains'] if
                                             domain['name'] == d['classification_domain']]
            else:
                classification_alphas = [float(domain['class_alpha']) for domain in d['domains'] if
                                         domain['name'] in d['selected_domains']]
                selected_domain_names_for_threshold = domain_names
            classification_dataset_path = create_classification_sum_tokens(reviews_to_similarities,
                                                                           selected_domain_names_for_threshold,
                                                                           classification_alphas, d[
                                                                               'classification_samples_per_domain'],
                                                                           d['seed'])
        elif d['classification_type'] == 'num_cat':
            if d.get('classification_domain'):
                selected_domain_names_for_threshold = [d['classification_domain']]
                classification_num_cats = [int(domain['class_num_cat']) for domain in d['domains'] if
                                             domain['name'] == d['classification_domain']]
            else:
                selected_domain_names_for_threshold = domain_names
                classification_num_cats = [int(domain['class_num_cat']) for domain in d['domains'] if
                                       domain['name'] in d['selected_domains']]
            classification_dataset_path = create_classification_num_of_top_categories(reviews_to_similarities,
                                                                                      selected_domain_names_for_threshold,
                                                                                      classification_num_cats,
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
        output_models = class_models  # + output_models
    if d.get('tasks'):
      tasks = d['tasks']
    elif d.get('in_domain',False):
        tasks = d['selected_domains']
    else:
        tasks = [f'{x[0]}_{x[1]}' for x in list(itertools.product(d['selected_domains'], repeat=2)) if x[0] != x[1]]
    d['server'] = socket.gethostname()
    d['training_file'] = training_file
    if d['masking'] != 'none':
        if d['masking'] == 'unigram':
            params = [domain['threshold'] for domain in d['domains'] if domain['name'] in d['selected_domains']]
        elif d['masking'] == 'bigram':
            params = [f'{domain["threshold"]}:{domain.get("bigram_threshold", domain["threshold"])}' for domain
                      in d['domains'] if domain['name'] in d['selected_domains']]
        d['masking_params'] = '_'.join([f'{y}-{x}' for x, y in zip(params, d['selected_domains'])])
    else:
        d['masking_params'] = ''
    d['num_ner_epochs'] = d.get('num_ner_epochs', 3)
    d['ner_dropout'] = d.get('ner_dropout', 0)
    d['f1_threshold'] = d.get('f1_threshold', 0.5)
    d['ae_lr'] = d.get('ae_lr', 5e-5)
    d["train_batch_size"] = d.get('train_batch_size', 32)
    run_benchmark(output_models, tasks, d['absa_dirs'], os.path.basename(training_file),
                                        d.get('gpu', True), d)
except Exception as e:
    logging.exception("error")

