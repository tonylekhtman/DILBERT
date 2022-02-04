# get the BertTokenizer as used in training and tokenize the long yelp and amazon texts
import argparse
import json
import logging
import os
import string

import numpy as np
from run_lm_finetuning import MODEL_CLASSES, load_and_cache_examples, TextDataset
from gensim.models import KeyedVectors
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords

more_words = ['still', 'might', 'enough', 'would']
stop_words = set(stopwords.words('english') + more_words)

from cat_config import get_cats


def get_embedding(word, model):
    if word not in model:
        return None
    word_vec = model[word]
    return word_vec


def calc_similarity(word1, word2, model):
    embedding_1 = get_embedding(word1, model)
    embedding_2 = get_embedding(word2, model)
    if embedding_1 is None or embedding_2 is None:
        return None
    return cosine_similarity([embedding_1], [embedding_2])


def calc_similarity_with_bigram(bigram, word, model):
    bigram_1, bigram_2 = bigram

    bigram_1_embedding = get_embedding(bigram_1, model)
    bigram_2_embedding = get_embedding(bigram_2, model)
    embedding_2 = get_embedding(word, model)
    if bigram_1_embedding is None or embedding_2 is None or bigram_2_embedding is None:
        return None
    embedding_1 = np.add(bigram_1_embedding, bigram_2_embedding)

    if embedding_1 is None or embedding_2 is None:
        return None
    return cosine_similarity([embedding_1], [embedding_2])


def get_max_cat_similarity(token_text, domain_categories, model, similarity_method):
    max_similarity = -1
    for domain_cat in domain_categories:
        similarity = similarity_method(token_text, domain_cat, model)
        if similarity is not None and similarity[0][0] > max_similarity:
            max_similarity = similarity[0][0].item()
    return max_similarity


def extract_bigram_max_similarities(dataset, embedding_model, domain_name, tokenizer, output):
    domain_categories = get_cats(domain_name)
    max_similarities = {}
    for example in tqdm(dataset.examples):
        new_example = []
        temp_token = []
        for token in reversed(example):
            token_text = tokenizer.ids_to_tokens[token]
            if token_text.startswith('##'):
                temp_token.append(token_text.strip('##'))
            else:
                temp_token.append(token_text)
                temp_token_str = ''.join(reversed(temp_token))
                new_example.append(temp_token_str)
                temp_token = []
        new_example = reversed(new_example)
        for bigram_token in ngrams(new_example, 2):
            token_1, token_2 = bigram_token
            if (':'.join([token_1, token_2])) in max_similarities:
                continue
            if token_1 in stop_words or token_2 in stop_words or token_1 in string.punctuation or token_2 in string.punctuation or token_1 in tokenizer.all_special_tokens or token_2 in tokenizer.all_special_tokens:
                max_similarities[':'.join([token_1, token_2])] = -1
            else:

                max_similarities[':'.join([token_1, token_2])] = get_max_cat_similarity(
                    (token_1, token_2),
                    domain_categories,
                    embedding_model,
                    calc_similarity_with_bigram)
    if not os.path.exists(os.path.dirname(output)):
        os.mkdir(os.path.dirname(output))
    json.dump(max_similarities, open(output, 'w+'))


def extract_unigram_max_similarities(dataset, model, name, tokenizer, output):
    domain_name = name
    domain_categories = get_cats(domain_name)
    max_similarities = {}
    for example in tqdm(dataset.examples):
        new_example = []
        temp_token = []
        for token in reversed(example):
            token_text = tokenizer.ids_to_tokens[token]
            if token_text.startswith('##'):
                temp_token.append(token_text.strip('##'))
            else:
                temp_token.append(token_text)
                temp_token_str = ''.join(reversed(temp_token))
                new_example.append(temp_token_str)
                temp_token = []
        new_example = reversed(new_example)
        for token in new_example:
            if token in max_similarities:
                continue
            if token in stop_words or token in string.punctuation:
                max_similarities[token] = -1
                continue
            max_similarities[token] = get_max_cat_similarity(token, domain_categories, model, calc_similarity)
    if not os.path.exists(os.path.dirname(output)):
        os.mkdir(os.path.dirname(output))
    json.dump(max_similarities, open(output, 'w+'))


max_sim_parser = argparse.ArgumentParser()
max_sim_parser.add_argument("--fasttext", action='store_true',
                            help="Which embeddings to use")
max_sim_parser.add_argument("--embedding_model", choices={'w2v', 'fasttext', 'custom_fasttext'}, default='fasttext')
max_sim_parser.add_argument("--embedding_model_path", type=str)
max_sim_parser.add_argument('--domain', type=str, choices={'rest', 'laptops', 'movies', 'hotel','devices','books'})
max_sim_parser.add_argument('--n', type=int, choices={1, 2}, default=1)
max_sim_parser.add_argument('--output', type=str)
max_sim_parser.add_argument('--path', type=str)


def create_index(args):
    current_domain = args.domain
    print(f'Creating similarity index for: {current_domain}')
    if args.fasttext or args.embedding_model == 'fasttext' or args.embedding_model == 'custom_fasttext':
        from gensim.models.wrappers import FastText

        if args.fasttext or args.embedding_model == 'fasttext':
            embedding_model = FastText.load_fasttext_format('embeddings/cc.en.300')
        else:
            embedding_model = FastText.load_fasttext_format(args.embedding_model_path)
    else:
        embedding_model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    # path, domain_name = DOMAIN_MAPPING[current_domain]
    path=args.path
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = TextDataset(tokenizer, file_path=path,
                          block_size=510)
    if args.n == 1:
        extract_unigram_max_similarities(dataset, embedding_model, current_domain, tokenizer, args.output)
    else:
        extract_bigram_max_similarities(dataset, embedding_model, current_domain, tokenizer, args.output)


if __name__ == '__main__':
    args = max_sim_parser.parse_args()
    create_index(args)
