import math
from typing import List

import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

"""只包含dae任务，fxl按照一个GitHub上的实现的"""


class GAIICDataCollator(object):
    def __init__(self, tokenizer, user_args):

        self.tokenizer = tokenizer

        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = [self.cls_token_id, self.sep_token_id]

        self.vocab_size = self.tokenizer.vocab_size

        # dae task
        self.mask_ratio = 0.3
        self.poisson_lambda = 3.5
        self.permutate_sentence_ratio = 1.0
        self.poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)

    def pad_and_truncate(self, input_ids_list, max_length):
        new_input_ids_list = []
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) > max_length:
                input_ids = input_ids[: max_length]
                input_ids = input_ids + [self.sep_token_id]
            elif len(input_ids) < max_length:
                pad = [self.pad_token_id] * (max_length - len(input_ids))
                input_ids = input_ids + pad
            else:
                input_ids = input_ids
            new_input_ids_list.append(input_ids)
        return new_input_ids_list

    # ============ text infilling ===========
    @staticmethod
    def _pad_sequence(tensors: List[torch.Tensor], padding_value: int, padding_side: str = 'right'):
        """
        Pad encoded inputs (on left/right and up to max length in the batch)
        """
        max_len = max(tensor.size(0) for tensor in tensors)
        padded_tensors = []
        if padding_side == 'right':
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        elif padding_side == 'left':
            for tensor in tensors:
                padding_length = max_len - len(tensor)
                padded_tensor = torch.cat([torch.full([padding_length], padding_value, dtype=tensor.dtype), tensor],
                                          dim=-1)
                padded_tensors.append(padded_tensor)
        padded_tensors = torch.stack(padded_tensors, dim=0)
        return padded_tensors

    def add_whole_word_mask(self, inputs):

        bsz, seq_len = inputs.size()
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # determine how many tokens we need to mask in total
        num_to_mask = math.ceil((~special_tokens_mask).sum() * self.mask_ratio)

        # generate a sufficient number of span lengths
        lengths = self.poisson_distribution.sample(sample_shape=(num_to_mask,))
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths, self.poisson_distribution.sample(sample_shape=(num_to_mask,))])
            cum_length = torch.cumsum(lengths, 0)

        # trim to about num_to_mask tokens
        idx = ((cum_length - num_to_mask) >= 0).nonzero()[0][0]
        lengths[idx] = num_to_mask - (0 if idx == 0 else cum_length[idx - 1])
        num_span = idx + 1
        lengths = lengths[:num_span]

        # handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_span - lengths.size(0)
        num_span -= num_inserts

        # select span start indices
        token_indices = (~special_tokens_mask).nonzero()
        rand_span = torch.randperm(token_indices.size(0))
        span_starts = rand_span[:num_span]

        # prepare mask and mask span start indices
        masked_indices = token_indices[span_starts]
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        mask[tuple(masked_indices.t())] = True
        lengths -= 1

        # fill up spans
        remaining = (lengths > 0) & (masked_indices[:, 1] < seq_len - 1)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            mask[tuple(masked_indices.t())] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < seq_len - 1)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask] = self.tokenizer.mask_token_id

        # remove mask tokens that are not starts of spans
        to_remove = mask & mask.roll(1, 1) | inputs.eq(self.tokenizer.pad_token_id)
        # calculate the number of inserted mask token per row
        inserts_num = torch.bincount(token_indices[rand_span[:num_inserts]][:, 0], minlength=bsz)
        new_inputs = []
        for i, example in enumerate(inputs):
            new_example = example[~to_remove[i]]
            n = inserts_num[i]
            if n:
                new_num = n + new_example.size(0)
                noise_mask = torch.zeros(new_num, dtype=torch.bool)
                mask_indices = torch.randperm(new_num - 2)[:n] + 1
                noise_mask[mask_indices] = 1
                result = torch.LongTensor(new_num.item())
                result[mask_indices] = self.tokenizer.mask_token_id
                result[~noise_mask] = new_example
                new_example = result
            new_inputs.append(new_example)

        new_inputs = self._pad_sequence(new_inputs, self.tokenizer.pad_token_id)

        return new_inputs

    # ============ text infilling ===========

    def build_inputs(self, input_ids_list, batch_input_ids_max_length):

        batch_use_decoders = []

        batch_decoder_input_ids, batch_origin_labels = [], []

        for input_ids in input_ids_list:
            tokens = input_ids
            tokens = torch.LongTensor(tokens)

            origin_labels = tokens.clone()
            origin_labels = torch.LongTensor(origin_labels)
            batch_origin_labels.append(origin_labels)

            use_decoder = 1

            batch_use_decoders.append(use_decoder)

        # text infilling
        text_infilled_ids = self.add_whole_word_mask(torch.tensor(input_ids_list, dtype=torch.long))
        batch_masked_input_ids = text_infilled_ids

        # set labels padding place to -100
        mlm_labels = torch.tensor([item for item in input_ids_list], dtype=torch.long)

        # build decoder input ids
        batch_decoder_input_ids = torch.tensor([item.cpu().numpy().tolist() for item in batch_origin_labels],
                                               dtype=torch.long)
        decoder_start_ids = torch.tensor([self.pad_token_id], dtype=torch.long)
        batch_decoder_start_ids = decoder_start_ids.unsqueeze(0).repeat_interleave(len(input_ids_list), dim=0)
        batch_decoder_input_ids = torch.cat([batch_decoder_start_ids, batch_decoder_input_ids[:, :-1]], dim=-1)

        # replace label's end token id to pad token id
        indexes = batch_decoder_input_ids == self.sep_token_id

        for i, sub_indexes in enumerate(indexes):
            idx = [j for j, item in enumerate(sub_indexes) if item == True]
            if len(idx) == 2:
                indexes[i][idx[0]] = False

        batch_decoder_input_ids[indexes] = self.pad_token_id

        # replace first token id to sep token id
        batch_decoder_input_ids[:, 0] = self.sep_token_id

        # to tensor
        batch_masked_input_ids = torch.tensor(batch_masked_input_ids,
                                              dtype=torch.long)
        batch_decoder_input_ids = torch.tensor([item.cpu().numpy().tolist() for item in batch_decoder_input_ids],
                                               dtype=torch.long)
        batch_lm_labels = mlm_labels
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

        batch_attention_mask = torch.gt(batch_masked_input_ids, 0)
        batch_decoder_attention_mask = torch.gt(batch_decoder_input_ids, 0)

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


