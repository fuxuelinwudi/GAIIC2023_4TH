import torch
from torch.nn.utils.rnn import pad_sequence
from pretrain.GAIICDataCollator.data_utils import text_segmentate, pseudo_summary_f1, get_input_mask,\
    padding_to_maxlength, shift_tokens_right


class GAIICDataCollator(object):
    def __init__(self, tokenizer, user_args):

        self.tokenizer = tokenizer

        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = [self.cls_token_id, self.sep_token_id]

        self.vocab_size = self.tokenizer.vocab_size

    @staticmethod
    def _pad_sequence(tensors, padding_value: int, padding_side: str = 'right'):
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

    def __call__(self, examples: list):

        labels = []
        attn_mask = []
        decoder_attn_mask = []
        source_inputs = []

        # batch_desc, batch_clinical, batch_dialog = list(zip(*examples))
        batch_desc, batch_clinical = list(zip(*examples))
        batch_max_length = max([len(desc_item.split(' ')) + len(clinical_item.split(' '))
                                for desc_item, clinical_item in zip(batch_desc, batch_clinical)])
        batch_input_text = [(desc_item, clinical_item) for desc_item, clinical_item in
                            zip(batch_desc, batch_clinical)]
        batch_use_decoders = []
        for text in batch_input_text:
            # todo，这里想加clinical或者dialog，可以拼接或者用10隔开
            texts = text[0]
            text = text_segmentate(texts)  # 对于本次数据

            if len(text) == 1 or len(text) == 0:
                continue
            use_decoder = 1
            if torch.rand(1).item() < 0.5:
                use_decoder = 0

            batch_use_decoders.append(use_decoder)

            sentence_id_vec, source, target, source_idxs, target_idxs = pseudo_summary_f1(
                text, '', self.tokenizer, batch_max_length + 3, "rouge-l")

            source_idxs, target_idxs = get_input_mask(sentence_id_vec, target_idxs)
            source_idxs[-1] = self.tokenizer.sep_token_id
            target_idxs = [self.cls_token_id] + target_idxs
            target_idxs[-1] = self.tokenizer.sep_token_id

            if len(source_idxs) > batch_max_length:
                if self.tokenizer.cls_token_id not in source_idxs[batch_max_length - 1:]:
                    source_idxs = source_idxs[:batch_max_length]
                    source_idxs[-1] = self.tokenizer.eos_token_id
                    print("Warning split long line: " + source + "\n")
                else:
                    continue

            source_idxs, attention_mask = padding_to_maxlength(
                source_idxs, batch_max_length, self.tokenizer.pad_token_id)

            label, target_attention_mask = padding_to_maxlength(
                target_idxs, batch_max_length, self.tokenizer.pad_token_id)

            # print("sample len: ", len(source_idxs))
            source_inputs.append(source_idxs)
            attn_mask.append(attention_mask)
            decoder_attn_mask.append(target_attention_mask)
            labels.append(label)

        labels = torch.tensor(labels)
        decode_input_idxs = shift_tokens_right(labels,
                                               self.tokenizer.pad_token_id,
                                               self.tokenizer.sep_token_id)

        input_ids = torch.tensor(source_inputs)
        attention_mask = torch.ne(input_ids, 0)
        decoder_input_ids = decode_input_idxs

        # replace label's end token id to pad token id
        indexes = decoder_input_ids == self.sep_token_id

        for i, sub_indexes in enumerate(indexes):
            idx = [j for j, item in enumerate(sub_indexes) if item == True]
            if len(idx) == 2:
                indexes[i][idx[0]] = False

        decoder_input_ids[indexes] = self.pad_token_id
        decoder_attention_mask = torch.ne(decoder_input_ids, 0)

        pad_indexes = labels == self.pad_token_id
        labels[pad_indexes] = -100

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels,
        #     "decoder_input_ids": decoder_input_ids,
        #     "decoder_attention_mask": decoder_attention_mask
        # }
        batch_use_decoders = torch.tensor(batch_use_decoders, dtype=torch.long)
        data_dict = {
            'input_ids': input_ids,
            # 'origin_input_ids': origin_input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'lm_labels': labels,
            'use_decoder': batch_use_decoders,
        }

        return data_dict