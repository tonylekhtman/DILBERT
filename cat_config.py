def get_cats(domain_name):
    with open(f'category_files/{domain_name}_categories.txt') as cat_file:
        domain_cats = [cat.strip() for cat in cat_file.readlines()]
    return domain_cats
