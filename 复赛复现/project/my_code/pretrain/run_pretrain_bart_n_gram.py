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
from modeling.modeling_bart import BartForPretrain, BartConfig, \
    BartForConditionalGeneration

from tricks.ema import EMA
from tricks.rdrop import compute_kl_loss
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

warnings.filterwarnings('ignore')

user_args = Config()
device = user_args.device
"""bart n-gram预训练
"""

parser = argparse.ArgumentParser()
parser.add_argument('--total_epoch', default=120, type=int, help='')
parser.add_argument('--batch_size', default=128, type=int, help='')
parser.add_argument('--pretrain_model_path', default='./pretrain/pretrain_model/new-bart-base', type=str, help='')
parser.add_argument('--save_path', default='./pretrain/pretrain_model/post-pretrained-bart-base-n-gram', type=str, help='')

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
        input_ids = tokenizer(desc).input_ids
        all_inputs['input_ids'].append(input_ids)

    semi_train_df = pd.read_csv(user_args.semi_train_path, header=None)
    for i, row in tqdm(semi_train_df.iterrows(), desc='Reading from semi train data', total=len(semi_train_df)):
        desc = row[1].strip()
        clinical = row[3]
        if str(clinical) != 'nan':
            input_ids = tokenizer(desc + ' ' + clinical).input_ids
        else:
            input_ids = tokenizer(desc).input_ids
        all_inputs['input_ids'].append(input_ids)

    test_a_df = pd.read_csv(user_args.test_a_path, header=None)
    for i, row in tqdm(test_a_df.iterrows(), desc='Reading from test a data', total=len(test_a_df)):
        desc = row[1].strip()
        input_ids = tokenizer(desc).input_ids
        all_inputs['input_ids'].append(input_ids)

    test_b_df = pd.read_csv(user_args.test_b_path, header=None)
    for i, row in tqdm(test_b_df.iterrows(), desc='Reading from test b data', total=len(test_b_df)):
        desc = row[1].strip()
        input_ids = tokenizer(desc).input_ids
        all_inputs['input_ids'].append(input_ids)

    return all_inputs


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


class GAIICDataCollator(object):
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = [self.cls_token_id, self.sep_token_id]

        self.vocab_size = self.tokenizer.vocab_size

        # mask ratio, bart use double
        self.max_ngram = 3
        self.mask_ratio = user_args.masked_lm_prob

    def pad_and_truncate(self, input_ids_list, max_length):
        new_input_ids_list = []
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) > max_length:
                input_ids = input_ids[: max_length - 1]
                input_ids = input_ids + [self.sep_token_id]
            elif len(input_ids) < max_length:
                pad = [self.pad_token_id] * (max_length - len(input_ids))
                input_ids = input_ids + pad
                input_ids[-1] = self.sep_token_id
            else:
                input_ids = input_ids
                input_ids[-1] = self.sep_token_id
            new_input_ids_list.append(input_ids)
        return new_input_ids_list

    # ============ mlm task ===================
    def _ngram_mask(self, input_ids, use_decoder):

        if use_decoder:
            mask_ratio = self.mask_ratio * 2
        else:
            mask_ratio = self.mask_ratio

        max_ngram = self.max_ngram

        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])
        num_to_predict = max(1, int(round(len(input_ids) * mask_ratio)))

        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)

        # shuffle probs
        random.shuffle(pvals)

        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        np.random.shuffle(ngram_indexes)

        covered_indexes = set()

        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels = torch.tensor(mask_labels, dtype=torch.long)

        return mask_labels

    def ngram_mask(self, input_ids_list, use_decoder_list):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, use_decoder_list[i])
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    # ============ mlm task ===================

    # ============ text infilling ===========
    def _text_infilling(self, src, p, mask_token_id):

        """
        src : -> torch.tensor
        find all none zero
        """
        src = [i for i in src if i != self.tokenizer.pad_token_id]
        src = torch.tensor(src, dtype=torch.long)

        is_word_start = torch.ones(src.size())
        num_to_mask = int(math.ceil(is_word_start.float().sum()) * p)  # 30% 마스킹

        if num_to_mask == 0:
            return src

        lambda_ = torch.ones(num_to_mask) + 2
        lengths = torch.poisson(lambda_)

        starting_words = is_word_start.nonzero(as_tuple=False)
        indices = starting_words[torch.randperm(starting_words.size(0))[:num_to_mask]].squeeze(1)

        # 1. zero-length masking
        zero_len_indices = [i for i in range(len(lengths)) if lengths[i] == 0]
        zero_mask_indices = []
        for i in zero_len_indices:
            zero_mask_indices.append(indices[i])
        zero_mask_indices.sort()

        src, lengths, indices, is_word_start = src.tolist(), lengths.tolist(), indices.tolist(), is_word_start.tolist()
        zero_len_loc = []
        for i, j in enumerate(zero_mask_indices):
            src.insert(i + j, mask_token_id)
            is_word_start.insert(i + j, 255)  # big integer for no more span masking
            zero_len_loc.append(j)

        # trim after zero span handling
        for i, j in enumerate(zero_len_indices):
            del lengths[j - i]
            del indices[j - i]

        zero_len_loc.sort()
        for i in range(len(indices)):
            for j in zero_len_loc:
                if indices[i] > j:
                    indices[i] += 1
                else:
                    break

        src, lengths, indices, is_word_start = torch.tensor(src), \
            torch.tensor(lengths), torch.tensor(indices), torch.tensor(is_word_start)

        num_to_mask = lengths.size(0)
        if num_to_mask == 0:
            return src
        assert (lengths > 0).all()

        # 2. initial masking (start indices)
        is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end of doc
        src[indices] = mask_token_id
        lengths -= is_word_start[indices]
        uncompleted = lengths > 0
        indices = indices[uncompleted]
        lengths = lengths[uncompleted]

        # 3. single masking
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            lengths = lengths[uncompleted]
            if indices.size(0) > 0:
                src[indices] = -100

        idx = 0
        while idx < len(src):
            if src[idx] == -100:
                assert src[idx - 1] == mask_token_id
                src = torch.cat([src[:idx], src[idx + 1:]])
                idx -= 1
            idx += 1

        return src

    def text_infilling(self, input_ids_list, use_decoder_list, default_p=0.3):
        infilled_input_ids_list = []
        for i, input_ids in enumerate(input_ids_list):
            if use_decoder_list[i] == 1:
                p = default_p
            else:
                p = default_p / 2
            infilled_input_ids = self._text_infilling(input_ids, p, self.mask_token_id)
            infilled_input_ids_list.append(infilled_input_ids)

        return infilled_input_ids_list

    # ============ text infilling ===========

    def mask_tokens(self, inputs, mask_labels):
        inputs = torch.tensor(inputs, dtype=torch.long)
        labels = inputs.clone()
        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=105, high=len(self.tokenizer), size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def build_inputs(self, input_ids_list, batch_input_ids_max_length):

        batch_use_decoders = []

        batch_decoder_input_ids, batch_origin_labels = [], []

        for input_ids in input_ids_list:
            tokens = input_ids
            tokens = torch.LongTensor(tokens)

            origin_labels = tokens.clone()
            origin_labels = torch.LongTensor(origin_labels)
            batch_origin_labels.append(origin_labels)

            decoder_input_ids = tokens.cpu().numpy().tolist()
            decoder_input_ids = [self.sep_token_id] + decoder_input_ids[: -1]
            decoder_input_ids = torch.LongTensor(decoder_input_ids)
            batch_decoder_input_ids.append(decoder_input_ids)

            use_decoder = 1
            if torch.rand(1).item() < 0.5:
                use_decoder = 0

            batch_use_decoders.append(use_decoder)

        # mask token finally, choose 80% to mask, 10% to random replace, 10% to keep
        batch_mask = self.ngram_mask(input_ids_list, batch_use_decoders)
        batch_masked_input_ids, mlm_labels = self.mask_tokens(input_ids_list, batch_mask)

        # """
        # dae, text infilling task
        # """
        # text_infilled_ids_list = self.text_infilling(input_ids_list, batch_use_decoders)
        # text_infilled_ids_list = [item.cpu().numpy().tolist() for item in text_infilled_ids_list]
        # text_infilled_ids_list = self.pad_and_truncate(text_infilled_ids_list, batch_input_ids_max_length)

        # to tensor
        batch_masked_input_ids = torch.tensor([item.cpu().numpy().tolist() for item in batch_masked_input_ids],
                                              dtype=torch.long)

        # batch_masked_input_ids = torch.tensor([item for item in text_infilled_ids_list],
        #                                       dtype=torch.long)

        batch_decoder_input_ids = torch.tensor([item.cpu().numpy().tolist() for item in batch_decoder_input_ids],
                                               dtype=torch.long)

        batch_lm_labels = mlm_labels

        # batch_lm_labels = torch.tensor([item for item in input_ids_list],
        #                                dtype=torch.long)

        batch_use_decoders = torch.tensor(batch_use_decoders, dtype=torch.long)

        return batch_masked_input_ids, batch_decoder_input_ids, \
            batch_lm_labels, batch_use_decoders

    def __call__(self, examples: list):

        batch_data = list(zip(*examples))
        origin_input_ids = batch_data[0]

        input_ids_list = [item for item in origin_input_ids]

        batch_input_ids_max_length = max([len(item) for item in origin_input_ids])
        input_ids_list = self.pad_and_truncate(input_ids_list, batch_input_ids_max_length)

        batch_masked_input_ids, batch_decoder_input_ids, \
            batch_lm_labels, batch_use_decoders = self.build_inputs(input_ids_list, batch_input_ids_max_length)

        batch_attention_mask = torch.ne(batch_masked_input_ids, self.pad_token_id).to(torch.float)
        batch_decoder_attention_mask = torch.ne(batch_decoder_input_ids, self.pad_token_id).to(torch.float)

        origin_input_ids = torch.tensor(input_ids_list, dtype=torch.long)

        data_dict = {
            'input_ids': batch_masked_input_ids,
            'origin_input_ids': origin_input_ids,
            'attention_mask': batch_attention_mask,
            'decoder_input_ids': batch_decoder_input_ids,
            'decoder_attention_mask': batch_decoder_attention_mask,
            'lm_labels': batch_lm_labels,
            'use_decoder': batch_use_decoders
        }

        return data_dict


def build_model():
    print(f'>>> Loading model : {user_args.pretrain_model_path}')
    config = BartConfig.from_pretrained(user_args.pretrain_model_path)
    language_model = BartForConditionalGeneration.from_pretrained(
        user_args.pretrain_model_path,
        config=config
    )
    model = BartForPretrain(
        config=config,
        language_model=language_model
    )
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    return model, config


def build_optimizer(model, total_steps, scheduler_type='linear'):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': user_args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
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
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    all_data_dict = read_data(tokenizer)
    train_dataset = GAIICDataset(all_data_dict)

    collator = GAIICDataCollator(tokenizer)

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
    warmup_steps = int(user_args.warmup_ratio * total_steps) + 1
    # warmup_steps = 1
    
    optimizer, scheduler = build_optimizer(model, total_steps, 'cosine')

    global_steps = 0
    cur_avg_loss = 0.
    model.zero_grad()
    scaler = GradScaler()
    for epoch in range(user_args.total_epoch):

        model.train()

        train_iterator = tqdm(train_data_loader, desc=f'Train epoch : {epoch + 1}',
                              total=len(train_data_loader))

        for index, data in enumerate(train_iterator):

            global_steps += 1
            
            if global_steps == warmup_steps:
                print('\nEMA starting ... ...')
                if user_args.ema:
                    ema = EMA(model.parameters(), decay=user_args.ema_weight_decay)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                lm_labels, use_decoder, origin_input_ids = \
                data['input_ids'], \
                    data['attention_mask'], \
                    data['decoder_input_ids'], \
                    data['decoder_attention_mask'], \
                    data['lm_labels'], \
                    data['use_decoder'], \
                    data['origin_input_ids']

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                lm_labels, use_decoder, origin_input_ids = \
                input_ids.to(device), \
                    attention_mask.to(device), \
                    decoder_input_ids.to(device), \
                    decoder_attention_mask.to(device), \
                    lm_labels.to(device), \
                    use_decoder.to(device),\
                    origin_input_ids.to(device)
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
            # loss.backward()
            scaler.scale(loss).backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            cur_avg_loss += loss.item()

            if user_args.ema:
                if global_steps >= warmup_steps:
                    ema.update(model.parameters())
                    
            scheduler.step()
            model.zero_grad()

            train_iterator.set_postfix(loss=loss.item(), global_step=global_steps)
        
        if user_args.ema:
            if global_steps >= warmup_steps:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
        
        if user_args.ema:
            if global_steps >= warmup_steps:
                ema.restore(model.parameters())
        
        epoch_avg_loss = cur_avg_loss / len(train_data_loader)
        cur_avg_loss = 0.

        print(f'\n>>> Epoch {epoch + 1}, avg loss : {epoch_avg_loss:.4f}')

    # """
    # save at last and evaluate
    # """
    model.language_model.save_pretrained(os.path.join(user_args.save_path, f'epoch_{user_args.total_epoch}'))
    tokenizer.save_pretrained(os.path.join(user_args.save_path, f'epoch_{user_args.total_epoch}'))

    print('>>> pretrain completed !!!')


def main():
    seed_everything(user_args.seed)
    pretrain()


if __name__ == '__main__':
    main()
