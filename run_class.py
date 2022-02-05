# read the in-domain files
# for each review get categoris above a certain threshold.
# set the labels above this threshold as the multi-label categories in a form of list.
# write the df to file
import os

import numpy as np
import torch
from sklearn import metrics
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import pandas as pd
import random

from torch import cuda

from benchmark import run_benchmark


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(model, training_loader, device, optimizer, scheduler, epoch):
    for _, data in enumerate(tqdm(training_loader), 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        model.train()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs[0], targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()


def validation(model, device, testing_loader, epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    tr_loss = 0
    global_step = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs[0], targets)
            tr_loss += loss.item()
            global_step += 1
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs[0]).cpu().detach().numpy().tolist())
    return fin_outputs, np.array(fin_targets), tr_loss/global_step


def run_classification_task(dataset_filename, num_of_epochs, pt_model_dir, output_dir, no_cuda=False):
    device = 'cuda' if cuda.is_available() and not no_cuda else 'cpu'
    df = pd.read_csv(dataset_filename)
    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['text', 'list']].copy()
    max_len = 200
    train_batch_size = 8
    valid_batch_size = 4
    epochs = num_of_epochs
    learning_rate = 5e-5
    seed = 200
    tokenizer = BertTokenizer.from_pretrained(pt_model_dir)
    train_size = 0.8
    train_dataset = new_df.sample(frac=train_size, random_state=seed).reset_index(drop=True)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    training_set = CustomDataset(train_dataset, tokenizer, max_len)
    testing_set = CustomDataset(test_dataset, tokenizer, max_len)
    train_params = {'batch_size': train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': valid_batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    num_of_classes = len(training_set.targets[0])
    config = BertConfig.from_pretrained(pt_model_dir, num_labels=num_of_classes)
    model = BertForSequenceClassification.from_pretrained(pt_model_dir, config=config)
    model.to(device)
    t_total = len(training_loader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    set_seed(seed)
    model.zero_grad()
    if os.path.exists(f'{output_dir}_{epochs-1}'):
        return 0, 0
    for epoch in range(epochs):
        train(model, training_loader, device, optimizer, scheduler, epoch)
        outputs, targets, loss = validation(model, device, testing_loader, epoch)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")

        outputs_train, targets_train, loss_train = validation(model, device, training_loader, epoch)
        outputs_train = np.array(outputs_train) >= 0.5
        accuracy = metrics.accuracy_score(targets_train, outputs_train)
        f1_score_micro = metrics.f1_score(targets_train, outputs_train, average='micro')
        f1_score_macro = metrics.f1_score(targets_train, outputs_train, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")


        # output_dir = f'class_model_1_sparse'
        output_dir_epoch = f'{output_dir}_{epoch}'
        if not os.path.exists(output_dir_epoch):
            os.makedirs(output_dir_epoch)
        model_to_save = model.module if hasattr(model,
                                                'module') else model
        model_to_save.save_pretrained(output_dir_epoch)
        tokenizer.save_pretrained(output_dir_epoch)
    return accuracy , loss


