import math
from collections import defaultdict

import json
import numpy as np
import random
from gensim.models.deprecated.fasttext_wrapper import FastText
from tqdm import tqdm
from nltk import sent_tokenize
from cat_config import get_cats
from max_sim import calc_similarity
import os
import pandas as pd

LAPTOPS_DOMAIN_NAME = 'laptops'
REST_DOMAIN_NAME = 'rest'
HOTEL_DOMAIN_NAME = 'hotel'
DEVICES_DOMAIN_NAME = 'devices'
BOOKS_DOMAIN_NAME = 'books'
domains = [LAPTOPS_DOMAIN_NAME, REST_DOMAIN_NAME]
cats = {d: get_cats(d) for d in domains}


def get_max_cat_similarity(token_text, domain_categories, model):
    max_similarity = -1
    max_cat = None
    for domain_cat in domain_categories:
        similarity = calc_similarity(token_text, domain_cat, model)
        if similarity is not None and similarity[0][0] > max_similarity:
            max_similarity = similarity[0][0].item()
            max_cat = domain_cat
    return max_similarity, max_cat


def create_dataset(reviews_path, domain, dataset_output_path, embedding_model_path='embeddings/cc.en.300', config={}):
    print(f'Creating the dataset cache for the CPP task for {domain}')
    if os.path.exists(dataset_output_path):
        return json.load(open(dataset_output_path, 'r+'))
    print(dataset_output_path)
    embedding_model = FastText.load_fasttext_format(embedding_model_path)
    with open(reviews_path, 'r') as reviews_fp:
        reviews = reviews_fp.readlines()
    reviews_res = []
    sampled_reviews = [y for x in random.sample(reviews, min(10000, len(reviews))) for y in sent_tokenize(x)]
    for review in tqdm(sampled_reviews):
        review_res = {}
        for cat in cats[domain]:
            review_res[cat] = {}
            for token in review.split():
                similarity = calc_similarity(token, cat, embedding_model)

                review_res[cat][token] = str(similarity[0][0]) if similarity is not None else str(-1)
        reviews_res.append((review_res, review))
    json.dump(reviews_res, open(dataset_output_path, 'w+'))
    return reviews_res


def create_classification_dataset_for_threshold(reviews_to_similarities, domains, thresholds,
                                                num_of_samples, seed=1):
    output_filename = f'{"_".join(domains)}_{"_".join([str(t) for t in thresholds])}_num_of_samples_{num_of_samples}.csv'
    if os.path.exists(output_filename):
        return output_filename
    res_dic = defaultdict(list)
    random.seed(seed)
    for domain, threshold in zip(domains, thresholds):
        for similarities, review in random.sample(reviews_to_similarities[domain], num_of_samples):
            res_dic['text'].append(review)
            for cat in cats[domain]:
                res_dic[f'{domain}_{cat}'].append(any(float(x) > threshold for x in similarities[cat].values()))
            for other_domain in [d for d in domains if d != domain]:
                for cat in cats[other_domain]:
                    res_dic[f'{other_domain}_{cat}'].append(False)
    df = pd.DataFrame(data=res_dic)
    df.to_csv(output_filename)
    return output_filename


def create_classification_dataset_dynamic(reviews_to_similarities, domain1, domain2, percent_of_categories,
                                          num_of_samples):
    output_filename = f'classification_datasets/cats%_{percent_of_categories}_num_of_samples_{num_of_samples}_{domain1}_{domain2}.csv'
    if os.path.exists(output_filename):
        return
    res_dic = defaultdict(list)
    random.seed(1)
    for similarities, review in random.sample(reviews_to_similarities[domain1], num_of_samples):
        res_dic['text'].append(review)
        cat_for_review = {}
        for cat in cats[domain1]:
            cat_for_review[cat] = max([float(x) for x in similarities[cat].values()])
        percentile = np.percentile(list(cat_for_review.values()), 100 - (percent_of_categories * 100))
        for cat in cats[domain1]:
            res_dic[f'{domain1}_{cat}'].append(any(float(x) > percentile for x in similarities[cat].values()))
        for cat in cats[domain2]:
            res_dic[f'{domain2}_{cat}'].append(False)
    random.seed(1)
    for similarities, review in random.sample(reviews_to_similarities[domain2], num_of_samples):
        res_dic['text'].append(review)
        cat_for_review = {}
        for cat in cats[domain2]:
            cat_for_review[cat] = max([float(x) for x in similarities[cat].values()])
        percentile = np.percentile(list(cat_for_review.values()), 100 - (percent_of_categories * 100))
        for cat in cats[domain2]:
            res_dic[f'{domain2}_{cat}'].append(any(float(x) > percentile for x in similarities[cat].values()))
        for cat in cats[domain1]:
            res_dic[f'{domain1}_{cat}'].append(False)
    df = pd.DataFrame(data=res_dic)
    df.to_csv(output_filename)


def create_classification_sum_tokens(reviews_to_similarities, domains, alphas,
                                     num_of_samples, seed):
    output_filename = f'classification_datasets/numeric_{seed}_{"_".join(domains)}_{"_".join([str(t) for t in alphas])}_samples_{num_of_samples}.csv'
    if os.path.exists(output_filename):
        return output_filename
    res_df = create_domain_df(domains, alphas, num_of_samples, reviews_to_similarities, seed)
    res_df = res_df[['text'] + sorted([c for c in res_df.columns if c != 'text'])]

    res_df.to_csv(output_filename)
    return output_filename


def create_domain_df(domains, alphas, num_of_samples, reviews_to_similarities, seed):
    res_dic = defaultdict(list)
    random.seed(seed)
    for domain, alpha in zip(domains, alphas):
        for similarities, review in random.sample(reviews_to_similarities[domain], num_of_samples):
            res_dic['text'].append(review)
            cat_for_review = {}
            for cat in cats[domain]:
                cat_for_review[cat] = sum([float(x) for x in similarities[cat].values()])
            num_of_categories_1 = round(alpha * len(review.split()))
            max_cats = [x[0] for x in
                        sorted(cat_for_review.items(), key=lambda x: x[1], reverse=True)[:num_of_categories_1]]
            for cat in cats[domain]:
                if cat in max_cats:
                    res_dic[f'{domain}_{cat}'].append(True)
                else:
                    res_dic[f'{domain}_{cat}'].append(False)
            for other_domain in [d for d in domains if d != domain]:
                for cat in cats[other_domain]:
                    res_dic[f'{other_domain}_{cat}'].append(False)
    df = pd.DataFrame(data=res_dic)
    return df


def create_classification_num_of_top_categories(reviews_to_similarities, domains, num_top_categories,
                                                num_of_samples, seed):
    output_filename = f'classification_datasets/num_cat_{seed}_{"_".join(domains)}_{"_".join([str(t) for t in num_top_categories])}_samples_{num_of_samples}.csv'
    if os.path.exists(output_filename):
        return output_filename
    res_dic = defaultdict(list)
    random.seed(seed)
    for domain, num_cat in zip(domains, num_top_categories):
        for similarities, review in random.sample(reviews_to_similarities[domain], num_of_samples):
            res_dic['text'].append(review)
            cat_for_review = {}
            for cat in cats[domain]:
                cat_for_review[cat] = sum([float(x) for x in similarities[cat].values()])
            max_cats = [x[0] for x in
                        sorted(cat_for_review.items(), key=lambda x: x[1], reverse=True)[:num_cat]]
            for cat in cats[domain]:
                if cat in max_cats:
                    res_dic[f'{domain}_{cat}'].append(True)
                else:
                    res_dic[f'{domain}_{cat}'].append(False)
            for other_domain in [d for d in domains if d != domain]:
                for cat in cats[other_domain]:
                    res_dic[f'{other_domain}_{cat}'].append(False)
    df = pd.DataFrame(data=res_dic)
    df = df[['text'] + sorted([c for c in df.columns if c != 'text'])]

    df.to_csv(output_filename)
    return output_filename


# thresholds = np.arange(0.31, 0.41, 0.02)
list_of_num_of_samples = [1000]
# thresholds1 = [0.32, 0.31, 0.32, 0.31, 0.315, 0.3, 0.315, 0.31]
# thresholds2 = [0.32, 0.32, 0.31, 0.31, 0.315, 0.3, 0.31, 0.315]
# for threshold1, threshold2 in tqdm(zip(thresholds1, thresholds2), desc='Thresholds'):
#     for num_of_samples in tqdm(list_of_num_of_samples, desc='Num of samples'):
#         create_classification_dataset_for_threshold(reviews_to_similarities,
#                                                     REST_DOMAIN_NAME,
#                                                     HOTEL_DOMAIN_NAME,
#                                                     threshold1, threshold2,
#                                                     num_of_samples)
# list_of_percent_of_categories = [0.1]  # [0.05, 0.1, 0.15, 0.2]
# for percent_of_categories in tqdm(list_of_percent_of_categories, desc='Percent of cats'):
#     for num_of_samples in tqdm(list_of_num_of_samples, desc='Num of samples'):
#         create_classification_dataset_dynamic(reviews_to_similarities, REST_DOMAIN_NAME, HOTEL_DOMAIN_NAME,
#                                               percent_of_categories,
#                                               num_of_samples)
if __name__ == '__main__':
    num_of_categories = [(1, 1), (2, 1), (2, 2)]  # [0.05, 0.1, 0.15, 0.2]
    alphas = [
        (0.03, 0.03),
        (0.04, 0.04),
        (0.05, 0.05)
        # (0.04, 0.04),
        # (0.05, 0.05), (0.05, 0.06), (0.06, 0.05), (0.06, 0.06), (0.04, 0.04), (0.03, 0.04), (0.04, 0.03),
        # (0.05, 0.04),
        # (0.04, 0.05), (0.03, 0.03)
    ]

    dfs = []
    for alpha in tqdm(alphas, desc='Percent of cats'):
        for num_of_samples in tqdm(list_of_num_of_samples, desc='Num of samples'):
            res = create_classification_sum_tokens(reviews_to_similarities, LAPTOPS_DOMAIN_NAME,
                                                   REST_DOMAIN_NAME,
                                                   alpha[0], alpha[1],
                                                   num_of_samples, 2)
            dfs.append(res)
    print(dfs)
