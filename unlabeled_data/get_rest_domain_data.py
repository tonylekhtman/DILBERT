import pandas as pd
import gzip


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def is_rest(data):
    if data['categories'] is None:
        return False
    if 'Restaurants' in data['categories']:
        return True
    return False


null = None

yelp_metadata_df = getDF('yelp_academic_dataset_business.json.gz')

only_rest_metadata_df = yelp_metadata_df[yelp_metadata_df.apply(is_rest, axis=1)]

yelp_review_df = getDF('yelp_academic_dataset_review.json.gz')

only_rest_review_df = yelp_review_df[yelp_review_df['business_id'].isin(only_rest_metadata_df['business_id'])]

only_rest_review_df.to_csv('only_rest_reviews.csv')

with open('rest.raw', 'w+') as yelp_reviews_fp:
    for review in only_rest_review_df.sample(240000):
        yelp_reviews_fp.write(review)
