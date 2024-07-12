import os
import torch
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from generation_wrapper import GenerationWrapper
from configurations.predict_config import Config

from transformers import BartForConditionalGeneration, BartConfig
from transformers import PegasusForConditionalGeneration, PegasusConfig
from modeling.modeling_cpt import CPTForConditionalGeneration, CPTConfig


warnings.filterwarnings('ignore')

user_args = Config()
device = user_args.device


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


def read_data(tokenizer, input_data_path) -> dict:
    test_a_df = pd.read_csv(input_data_path, header=None)

    inputs = defaultdict(list)
    for i, row in tqdm(test_a_df.iterrows(), desc='Reading from test a data', total=len(test_a_df)):

        # if i == 7500:
        #     break

        desc = row[1].strip()
        clinical = row[2]
        if str(clinical) == 'nan':
            clinical = tokenizer.unk_token
        inputs['desc'].append(desc)
        inputs['clinical'].append(clinical)
    return inputs


class GAIICDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['desc'][index],
            self.data_dict['clinical'][index]
        )
        return data

    def __len__(self) -> int:
        return len(self.data_dict['desc'])


class GAIICDataCollator(object):
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, examples: list):

        batch_desc, batch_clinical = list(zip(*examples))
        batch_max_length = max([len(desc_item.split(' ')) + len(clinical_item.split(' '))
                                for desc_item, clinical_item in zip(batch_desc, batch_clinical)])
        batch_input_text = [(desc_item, clinical_item) for desc_item, clinical_item in
                            zip(batch_desc, batch_clinical)]
        batch_encode_outputs = self.tokenizer.batch_encode_plus(batch_input_text,
                                                                add_special_tokens=True,
                                                                max_length=batch_max_length,
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors='pt',
                                                                return_token_type_ids=False)

        data_dict = {
            'input_ids': batch_encode_outputs['input_ids'],
            'attention_mask': batch_encode_outputs['attention_mask']
        }

        return data_dict


def build_bart_model(finetuned_model_path, is_half=True):
    print(f'>>> Loading bart model from : {finetuned_model_path}')
    config = BartConfig.from_pretrained(finetuned_model_path)
    model = BartForConditionalGeneration(config=config)
    if is_half:
        model.half()
    model.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'pytorch_model.bin'),
                                     map_location=device))

    model.to(device)
    return model


def build_cpt_model(finetuned_model_path, is_half=True):
    print(f'>>> Loading cpt model from : {finetuned_model_path}')
    config = CPTConfig.from_pretrained(finetuned_model_path)
    model = CPTForConditionalGeneration(config=config)
    if is_half:
        model.half()
    model.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'pytorch_model.bin'),
                                     map_location=device))

    model.to(device)
    return model


def build_pegasus_model(finetuned_model_path, is_half=True):
    print(f'>>> Loading pegasus model from : {finetuned_model_path}')
    config = PegasusConfig.from_pretrained(finetuned_model_path)
    model = PegasusForConditionalGeneration(config=config)
    if is_half:
        model.half()
    model.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'pytorch_model.bin'),
                                     map_location=device))

    model.to(device)
    return model


def clean(decoder_output):
    bad_tokens = ['[CLS]', '[SEP]', '[PAD]']
    new_decoder_output = []
    for token in decoder_output:
        new_decoder_output.append(token.replace(' ', ''))
    new_decoder_output1 = []
    for token in new_decoder_output:
        if token not in bad_tokens:
            new_decoder_output1.append(token)
    return new_decoder_output1


def prepare_inputs(inputs, device='cuda'):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def predict(model, tokenizer, loader):

    res = []

    test_iterator = tqdm(loader, desc='Prediction', total=len(loader))
    for index, data in enumerate(test_iterator):

        data = prepare_inputs(data)

        with torch.no_grad():
            outputs = model.generate(
                **data,
                num_beams=user_args.num_beams,
                min_length=user_args.min_length,
                max_length=user_args.max_length,
                early_stopping=user_args.early_stopping,
                no_repeat_ngram_size=user_args.no_repeat_ngram_size
            )

        batch_predictions = []
        for i in range(len(outputs)):
            decoded_preds = tokenizer.batch_decode(
                outputs[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_preds = clean(decoded_preds)
            batch_predictions.append(' '.join(decoded_preds))

        res.extend(batch_predictions)

    return res


def invoke(input_data_path, output_data_path):

    seed_everything(user_args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(user_args.finetuned_model_path1)

    model1 = build_bart_model(user_args.finetuned_model_path1, False)
    model1.eval()

    model2 = build_bart_model(user_args.finetuned_model_path2, False)
    model2.eval()

    model3 = build_bart_model(user_args.finetuned_model_path3)
    model3.eval()

    model4 = build_bart_model(user_args.finetuned_model_path4)
    model4.eval()

    model5 = build_pegasus_model(user_args.finetuned_model_path5)
    model5.eval()

    model6 = build_pegasus_model(user_args.finetuned_model_path6)
    model6.eval()

    model7 = build_cpt_model(user_args.finetuned_model_path7, False)
    model7.eval()

    model8 = build_cpt_model(user_args.finetuned_model_path8, False)
    model8.eval()

    model9 = build_cpt_model(user_args.finetuned_model_path9)
    model9.eval()

    model = GenerationWrapper(
        model_list=[model1, model2, model3, model4, model5, model6,
                    model7, model8, model9],
        ratio_list=[0.225, 0.225, 0.125, 0.125, 0.15, 0.15, 0.35, 0.35, 0.3]
    )

    data_dict = read_data(tokenizer, input_data_path)
    dataset = GAIICDataset(data_dict)
    collator = GAIICDataCollator(tokenizer)

    test_data_loader = DataLoader(
        dataset,
        batch_size=user_args.batch_size,
        num_workers=user_args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    indexes = list(range(len(dataset)))
    predict_res = predict(model, tokenizer, test_data_loader)

    with open(output_data_path, 'w', encoding='utf-8') as f:
        for i in range(len(predict_res)):
            f.writelines(str(indexes[i]) + ',' + predict_res[i] + '\n')


if __name__ == '__main__':
    # invoke('../data/semi_train.csv', 'ou.txt')
    pass

