# DILBERT

### Official code repository for the EMNLP'21 paper - ["DILBERT: Customized Pre-Training for Domain Adaptation with Category Shift, with an Application to Aspect Extraction"](https://aclanthology.org/2021.emnlp-main.20.pdf)

DILBERT is a Pre-training method for Domain Adaptaion. 
DILBERT is especially suitable for tasks with Category Shift. In this work we mainly dealt with the challenging task of Aspect Extraction.
DILBERT's pre-training is based on unlabelled-data from the domains you are trying to adapt between.


### Usage instructions

Running an experiment with DILBERT consists of the following steps:
1. Obtaining categories for your source domain and target domain
2. Obtaining unlabled data for your source domain and target domain
3. [Optional] Creating custom word embeddings for the combination of your unlbaled data.
4. Pre-training with the CMLM task (Categorical Masked Language Model) using the unlabeled data and the categories.
5. Pre-training with the CPP task (Category Proxy Prediction) using the unlabeled data and the categories.
6. Fine-tuning with the Aspect Extraction task on the source domain labeled data.
7. Applying the model on the target domain data to get extracted aspects.


### 0. Setup a conda environment
You can run the following command to create a conda environment from our .yml file:
```
conda env create --file dilbert_env.yml
conda activate absada
```

### 1. Obtain categories for your source and target domains
Under the ```category_files``` directory create a file named ``<domain>_categories.txt``

You can see the files: `category_files/laptops_categories.txt` and `category_files/rest_categories.txt` 

### 2. Obtain unlabeled data for your source and target domain
In this work we've obtained the unlabeled data from  Amazon and Yelp.

To get the restaurants unlabeled data you can go to: https://www.yelp.com/dataset

Put the files: yelp_academic_dataset_business.json.gz and yelp_academic_dataset_review.json.gz in the `unlabeled_data` dir and then:
```
cd unlabeled_data
python get_rest_domain_data.py
```


To get the laptops unlabeled data you can go to: https://nijianmo.github.io/amazon/index.html.
Put the files: meta_Electronics.json.gz and Electronics.json.gz in the `unlabeled_data` dir and then:
```
cd unlabeled_data
python get_laptop_domain_data.py
```

### 3. Obtain data for your downstream task
In the `ae_files` you can create a directory for your domain.

The files in the directory are in the BIO style, for example:
```
But O
the O
staff B-AS
was O
so O
horrible O
to O
us O
. O
```
There are already directories for: rest, laptops and mams datasets.

### 4. Run a training
The `main.py` script runs the whole process.

It gets a training configuration as a parameter.

There is an example in `trainings/sample.yaml`

```
absa_dirs:
- ae_files
classification: true
classification_domain: laptops
classification_epochs: 1
classification_samples_per_domain: 2000
mlm_thresholds:
  rest: 90
  laptops: 90
classification_thresholds:
  rest: 0.3
  laptops: 0.3
masking: unigram
pre_trained_model_name_or_path: bert-base-uncased
model_type: bert
num_ner_epochs: 3
num_cmlm_epochs: 1
output_path: dilbert-v1-new2
save_steps: 0
seed: 1948
selected_domains:
- laptops
- rest
embedding_model: custom_fasttext
tasks:
  - laptops
```
absa_dirs - The dir where the domain directories exist.

classification - Whether to run the CPP step.

classification_domain - on which domain to run the CPP.

classification_epochs - Number of epochs for CPP.

classification_samples_per_domain - Number of samples for CPP.

mlm_thresholds - What percent of tokens to keep unmasked.

classification_thresholds - What is the similarity threshold for the CPP task.

masking - Whether to run CMLM, `none` for no CMLM, `unigram` for CMLM

model_type - What kind of transformer to use. Tested on bert.

pre_trained_model_name_or_path - What subtype of the model_type to use. Tested on bert-base-uncased

num_ner_epochs - number of epochs for the fine-tune on Aspect Extraction.

num_cmlm_epochs - number of epochs for the cmlm pre-training.

output_path - The output name for the cmlm model.

selected_domains - The domains to use in the pre-training steps: CMLM and CPP.

embedding_model - Whether to use custom embeddings or not. Options are custom_fasttext or fasttext.

tasks - Which setups to run. For example: `rest_laptop` for rest to laptop, `rest` for in domain setup.


To tun the full process you can run:
```
python main.py --training_file trainins/sample.yaml --gpu 0
```

training_file - A yaml that sets the experiment
gpu - On which device to run







