# coding:utf-8
import argparse
import os
import math
import torch
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, BertTokenizer
from torch.cuda.amp import autocast, GradScaler
from pretrain_config import Config
import sys
sys.path.append('.')
from modeling.modeling_cpt import CPTForPretrain, CPTConfig, \
    CPTForConditionalGeneration
from pretrain_evaluate import Evaluate
from GAIICDataCollator.DataCollatorV4 import GAIICDataCollator

warnings.filterwarnings('ignore')

user_args = Config()
device = user_args.device
"""cpt dae预训练
"""
parser = argparse.ArgumentParser()
parser.add_argument('--total_epoch', default=120, type=int, help='')
parser.add_argument('--batch_size', default=128, type=int, help='')
parser.add_argument('--pretrain_model_path', default='./pretrain/pretrain_model/new-cpt-base', type=str, help='')
parser.add_argument('--save_path', default='./pretrain/pretrain_model/post-pretrained-cpt-base-dae', type=str, help='')

args = parser.parse_args()
user_args.pretrain_model_path = args.pretrain_model_path
user_args.total_epoch = args.total_epoch
user_args.save_path = args.save_path
user_args.batch_size = args.batch_size
if not os.path.exists(user_args.save_path):
    os.mkdir(user_args.save_path)
print(user_args.total_epoch)


def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def read_data(tokenizer):
    all_inputs = defaultdict(list)

    train_df = pd.read_csv(user_args.train_path, header=None)
    for i, row in tqdm(train_df.iterrows(), desc='Reading from train data', total=len(train_df)):
        desc = row[1].strip()
        clinical = tokenizer.unk_token
        input_ids = tokenizer(desc, clinical).input_ids
        all_inputs['input_ids'].append(input_ids)

    semi_train_df = pd.read_csv(user_args.semi_train_path, header=None)
    for i, row in tqdm(semi_train_df.iterrows(), desc='Reading from semi train data', total=len(semi_train_df)):
        desc = row[1].strip()
        clinical = row[3]
        if str(clinical) != 'nan':
            input_ids = tokenizer(desc, clinical).input_ids
        else:
            clinical = tokenizer.unk_token
            input_ids = tokenizer(desc, clinical).input_ids
        all_inputs['input_ids'].append(input_ids)

    test_a_df = pd.read_csv(user_args.test_a_path, header=None)
    for i, row in tqdm(test_a_df.iterrows(), desc='Reading from test a data', total=len(test_a_df)):
        desc = row[1].strip()
        clinical = tokenizer.unk_token
        input_ids = tokenizer(desc, clinical).input_ids
        all_inputs['input_ids'].append(input_ids)

    test_b_df = pd.read_csv(user_args.test_b_path, header=None)
    for i, row in tqdm(test_b_df.iterrows(), desc='Reading from test b data', total=len(test_b_df)):
        desc = row[1].strip()
        clinical = tokenizer.unk_token
        input_ids = tokenizer(desc, clinical).input_ids
        all_inputs['input_ids'].append(input_ids)
    print('>>> all data num: ', len(all_inputs['input_ids']))
    return all_inputs, None, None


class GAIICDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
        )
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])



def build_model():
    print(f'>>> Loading model : {user_args.pretrain_model_path}')
    config = CPTConfig.from_pretrained(user_args.pretrain_model_path)
    language_model = CPTForConditionalGeneration.from_pretrained(
        user_args.pretrain_model_path,
        config=config
    )
    model = CPTForPretrain(
        config=config,
        language_model=language_model,
        ignore_loss_index=0,
    )

    model.to(device)
    return model, config


def build_optimizer(model, total_steps, scheduler_type='linear'):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': user_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=user_args.learning_rate,
                      weight_decay=user_args.weight_decay, eps=user_args.eps)

    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(user_args.warmup_ratio * total_steps) + 1,
                                                    num_training_steps=total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(user_args.warmup_ratio * total_steps) + 1,
                                                    num_training_steps=total_steps)

    return optimizer, scheduler


def pretrain():
    tokenizer = BertTokenizer.from_pretrained(user_args.pretrain_model_path)
    model, config = build_model()

    all_data_dict, train_data_dict, val_data_dict, = read_data(tokenizer)

    valid_dataset = None
    if user_args.pretrain_with_dev:
        train_dataset = GAIICDataset(train_data_dict)
        valid_dataset = GAIICDataset(val_data_dict)
    else:
        train_dataset = GAIICDataset(all_data_dict)

    collator = GAIICDataCollator(tokenizer, user_args)

    valid_data_loader = None
    eval_process = None
    if user_args.pretrain_with_dev:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=user_args.batch_size,
            num_workers=user_args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )

        eval_process = Evaluate(tokenizer, device,
                                max_ans_length=user_args.encoder_max_length,
                                num_beams=5)

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=user_args.batch_size,
            num_workers=user_args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    else:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=user_args.batch_size,
            num_workers=user_args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )

    total_steps = len(train_data_loader) * user_args.total_epoch

    optimizer, scheduler = build_optimizer(model, total_steps, 'cosine')

    global_steps = 0
    cur_avg_loss = 0.
    best_cider_with_epoch = {'epoch': 0, 'cider': 0.0}
    model.zero_grad()
    scaler = GradScaler()
    for epoch in range(user_args.total_epoch):

        model.train()

        train_iterator = tqdm(train_data_loader, desc=f'Train epoch : {epoch + 1}',
                              total=len(train_data_loader))

        for index, data in enumerate(train_iterator):

            global_steps += 1

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels, use_decoder = \
                data['input_ids'], \
                    data['attention_mask'], \
                    data['decoder_input_ids'], \
                    data['decoder_attention_mask'], \
                    data['lm_labels'], \
                    data['use_decoder']

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels, use_decoder = \
                input_ids.to(device), \
                    attention_mask.to(device), \
                    decoder_input_ids.to(device), \
                    decoder_attention_mask.to(device), \
                    lm_labels.to(device), \
                    use_decoder.to(device)
            with autocast():
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    lm_labels=lm_labels,
                    use_decoder=use_decoder
                )
            # loss = torch.mean(loss)
            scaler.scale(loss).backward()
            # loss.backward()

            cur_avg_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            model.zero_grad()

            train_iterator.set_postfix(loss=loss.item(), global_step=global_steps)

        epoch_avg_loss = cur_avg_loss / len(train_data_loader)
        cur_avg_loss = 0.
        print(f'\n>>> Epoch {epoch + 1}, average loss : {epoch_avg_loss:.4f}')

    """
    save at last and evaluate
    """
    model.language_model.save_pretrained(os.path.join(user_args.save_path, f'epoch_{user_args.total_epoch}'))
    tokenizer.save_pretrained(os.path.join(user_args.save_path, f'epoch_{user_args.total_epoch}'))

    if user_args.pretrain_with_dev:
        model.eval()
        predictions, references = eval_process(model.language_model, valid_data_loader)
        ciderD_f1 = eval_process.compute_ciderD(predictions, references)
        print(f'\n>>> Epoch: @@{user_args.total_epoch}@@, test avg ciderD F1: @@{ciderD_f1:.4f}@@')
        model.train()

    print('>>> pretrain completed !!!')


def main():
    seed_everything(user_args.seed)
    pretrain()


if __name__ == '__main__':
    main()
