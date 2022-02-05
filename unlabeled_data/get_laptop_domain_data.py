import pandas as pd
import gzip

from tqdm import tqdm

relevant_cats = ['Laptops', 'laptops']
metadata_file = 'meta_Electronics.json.gz'
reviews_file = 'Electronics.json.gz'
cached_file = 'only_laptop_reviews.csv'
output_file = 'laptops.raw'


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:

        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in tqdm(parse(path)):
        if i == 220000:
            break
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')




def is_category_laptop(row):
    for x in relevant_cats:
        # if x in row['category'] :
        if x in row['categories'][0]:
            return True
    return False


meta_Electronics_df = getDF(metadata_file)

true = "true"
false = "false"
review_electronics_df = getDF(reviews_file)

only_laptops_meta_df = meta_Electronics_df[meta_Electronics_df.apply(is_category_laptop, axis=1)]

only_laptops_review_df = review_electronics_df[review_electronics_df['asin'].isin(only_laptops_meta_df['asin'])]
only_laptop_review_texts = only_laptops_review_df['reviewText']
only_laptops_review_df.to_csv(cached_file)

with open(output_file, 'w+') as amazon_laptop_reviews_fp:
    for i, review_text in enumerate(only_laptop_review_texts):
        if type(review_text) == float:
            continue
        amazon_laptop_reviews_fp.write(review_text)
        amazon_laptop_reviews_fp.write('\n')
