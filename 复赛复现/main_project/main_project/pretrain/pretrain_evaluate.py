from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import os
from six.moves import cPickle
import math
import six


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)


def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref, n=4, sigma=6.0):
    '''
    Compute the cosine similarity of two vectors.
    :param vec_hyp: array of dictionary for vector corresponding to hypothesis
    :param vec_ref: array of dictionary for vector corresponding to reference
    :param norm_hyp: array of float for vector corresponding to hypothesis
    :param norm_ref: array of float for vector corresponding to reference
    :param length_hyp: int containing length of hypothesis
    :param length_ref: int containing length of reference
    :return: array of score for each n-grams cosine similarity
    '''
    delta = float(length_hyp - length_ref)
    # measure consine similarity
    val = np.array([0.0 for _ in range(n)])
    for n in range(n):
        # ngram
        for (ngram, count) in vec_hyp[n].items():
            # vrama91 : added clipping
            val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

        if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
            val[n] /= (norm_hyp[n] * norm_ref[n])

        assert (not math.isnan(val[n]))
        # vrama91: added a length based gaussian penalty
        # print('penalty', length_hyp, length_ref, np.e**(-(delta**2)/(2*sigma**2)))
        val[n] *= np.e ** (-(delta ** 2) / (2 * sigma ** 2))
    return val


class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def copy_empty(self):
        new = CiderScorer(df_mode="corpus", n=self.n, sigma=self.sigma)
        new.df_mode = self.df_mode
        new.ref_len = self.ref_len
        new.document_frequency = self.document_frequency
        return new

    def __init__(self, df_mode="corpus", test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        self.ref_len = None
        self.document_frequency = defaultdict(float)
        if self.df_mode != "corpus":
            pkl_file = cPickle.load(open(os.path.join(df_mode), 'rb'), **(dict(encoding='latin1') if six.PY3 else {}))
            self.ref_len = np.log(float(pkl_file['ref_len']))
            self.document_frequency = pkl_file['document_frequency']
        self.cook_append(test, refs)

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                self.ctest.append(cook_test(test, self.n))  ## N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def counts2vec(self, cnts):
        """
        Function maps counts of ngram to vector of tfidf weights.
        The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
        The n-th entry of array denotes length of n-grams.
        :param cnts:
        :return: vec (array of dict), norm (array of float), length (int)
        """
        vec = [defaultdict(float) for _ in range(self.n)]
        length = 0
        norm = [0.0 for _ in range(self.n)]
        for (ngram, term_freq) in cnts.items():
            # give word count 1 if it doesn't appear in reference corpus
            df = np.log(max(1.0, self.document_frequency[ngram]))
            # ngram index
            n = len(ngram) - 1
            # tf (term_freq) * idf (precomputed idf) for n-grams
            vec[n][ngram] = float(term_freq) * (self.ref_len - df)
            # compute norm for the vector.  the norm will be used for computing similarity
            norm[n] += pow(vec[n][ngram], 2)

            if n == 1:
                length += term_freq
        norm = [np.sqrt(n) for n in norm]
        return vec, norm, length

    def compute_cider(self):

        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))
        # elif self.df_mode == "coco-val-df":
        # if coco option selected, use length of coco-val set
        #    self.ref_len = np.log(float(40504))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = self.counts2vec(test)
            # compute vector for ref captions
            score = np.zeros((len(refs), self.n))
            for rid, ref in enumerate(refs):
                vec_ref, norm_ref, length_ref = self.counts2vec(ref)
                score[rid] += sim(vec, vec_ref, norm, norm_ref, length, length_ref, self.n, self.sigma)
            # print(score)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score, 1)  # Cider本身就是从1gram到ngram的平均值。
            # divide by number of references
            score_avg = np.sum(score_avg) / len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        if self.df_mode == "corpus":
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            # assert to check document frequency
            assert (len(self.ctest) >= max(self.document_frequency.values()))
            # import json for now and write the corresponding files
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)

    def my_get_cider(self, gts, res):

        crefs = [precook(_, self.n) for _ in gts]
        ctest = [precook(_, self.n) for _ in res]

        assert self.ref_len is not None

        scores = np.zeros((len(ctest), len(crefs), self.n))

        for tid, test in enumerate(ctest):
            vec, norm, length = self.counts2vec(test)
            for rid, ref in enumerate(crefs):
                vec_ref, norm_ref, length_ref = self.counts2vec(ref)
                scores[tid, rid] += sim(vec, vec_ref, norm, norm_ref, length, length_ref, self.n, self.sigma)

        scores = np.mean(scores, -1)
        scores *= 10.0

        return scores

    def my_get_self_cider(self, res):

        ctest = [self.counts2vec(precook(_, self.n)) for _ in res]

        assert self.ref_len is not None

        scores = np.zeros((len(res), len(res), self.n))

        for tid, test in enumerate(ctest):
            vec, norm, length = test
            for rid, ref in enumerate(ctest):
                vec_ref, norm_ref, length_ref = ref
                scores[tid, rid] += sim(vec, vec_ref, norm, norm_ref, length, length_ref, self.n, self.sigma)

        scores = np.mean(scores, -1)
        scores *= 10.0

        return scores


class CiderD:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, n=4, sigma=6.0, df="corpus"):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        # set which where to compute document frequencies from
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, sigma=sigma, df_mode=self._df)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)
            tmp_cider_scorer += (hypo[0], ref)

        (score, scores) = tmp_cider_scorer.compute_score()

        return score, scores

    def my_compute_score(self, gts, res, avg_refs=True):
        """
        res a list of list
        gts a list of list
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()

        scores = []
        for _gts, _res in zip(gts, res):

            tmp = tmp_cider_scorer.my_get_cider(_gts, _res)
            if avg_refs:
                tmp = np.mean(tmp, 1)
            else:
                tmp = np.mean(tmp, 1)
            scores.append(tmp)
        scores = np.array(scores)
        score = np.mean(scores)
        return score, scores

    def my_self_cider(self, res):
        """
        gts a list of list
        """
        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()
        scores = []
        for _res in res:
            tmp = tmp_cider_scorer.my_get_self_cider(_res)
            scores.append(tmp)
        return scores

    def method(self):
        return "CIDEr-D"


class Smoother():
    def __init__(self, window):
        self.window = window
        self.num = {}
        self.sum = {}

    def update(self, **kwargs):
        """
        为了调用方便一致，支持kwargs中有值为None的，会被忽略
        kwargs中一些值甚至可以为dict，也就是再套一层。
        示例: update(a=1, b=2, c={'c':1, 'd':3})，相当于update(a=1, b=2, c=1, d=3)
        如果值为参数的None的话忽略
        """
        values = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    values[x] = kwargs[key][x]  # 有可能会覆盖，如update(a=1,b={'a':2})
            else:
                values[key] = kwargs[key]
        for key in values:
            if values[key] is None:
                continue
            if key not in self.num:
                self.num[key] = []
                self.sum[key] = 0
            self.num[key].append(values[key])
            self.sum[key] += values[key]

            if len(self.num[key]) > self.window:
                self.sum[key] -= self.num[key][-self.window - 1]
            if len(self.num[key]) > self.window * 2:
                self.clear(key)
        pass

    def clear(self, key):
        del self.num[key][:-self.window]

    def value(self, key=None, mean=True):
        if mean:
            if key is None:
                return {key: self.sum[key] / min(len(self.num[key]), self.window) for key in self.num}
            return self.sum[key] / min(len(self.num[key]), self.window)
        if key is None:
            return {key: np.array(self.num[key]) for key in self.num}
        return np.array(self.sum[key])

    def keys(self):
        return self.num.keys()


class Evaluate(object):
    def __init__(self, tokenizer, device, max_ans_length=60, num_beams=3):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_ans_length
        self.num_beams = num_beams

    def compute_ciderD(self, preds, trues):
        metrics = Smoother(100)
        res, gts = [], {}
        tot = 0
        for i in range(len(preds)):
            res.append({'image_id': tot, 'caption': [str(preds[i])]})
            gts[tot] = [str(trues[i])]
            tot += 1
        CiderD_scorer = CiderD(df='corpus', sigma=15)
        cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        metrics.update(cider=cider_score)
        score = metrics.value()['cider']
        return score

    def __call__(self, model, val_loader):
        predictions = []
        references = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='validation: '):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = torch.gt(input_ids, 0).to(self.device)
                target_ids = batch['origin_input_ids'].to(self.device)

                outputs = model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         max_length=self.max_length,
                                         num_beams=self.num_beams,
                                         early_stopping=True)

                # 解码输出结果和参考结果
                decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_refs = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_refs)

        return predictions, references


class GAIICDevDataCollator(object):
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def pad_and_truncate(self, input_ids_list, max_length):
        new_input_ids_list = []
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) > max_length:
                input_ids = input_ids[: max_length - 1]
                input_ids = input_ids + [self.sep_token_id]
            elif len(input_ids) < max_length:
                pad = [self.pad_token_id] * (max_length - len(input_ids) - 1) + [self.sep_token_id]
                input_ids = input_ids + pad
            else:
                input_ids = input_ids
            new_input_ids_list.append(input_ids)
        return new_input_ids_list

    def __call__(self, examples: list):

        batch_data = list(zip(*examples))
        origin_input_ids = batch_data[0]
        input_ids_list = [item.cpu().numpy().tolist() for item in origin_input_ids]

        batch_input_ids_max_length = max([len(item) for item in origin_input_ids])
        input_ids_list = self.pad_and_truncate(input_ids_list, batch_input_ids_max_length)

        batch_source = torch.tensor(input_ids_list,
                                    dtype=torch.long)

        data_dict = {
            'input_ids': batch_source,
        }

        return data_dict


def read_dev_data(tokenizer, path) -> dict:
    dev_df = pd.read_csv(path, header=None)

    inputs = defaultdict(list)
    for i, row in tqdm(dev_df.iterrows(), desc='Reading from pretraining with dev data', total=len(dev_df)):
        desc = row[1].strip()
        input_ids = tokenizer(desc, return_tensors="pt").input_ids
        inputs['input_ids'].append(input_ids)

    return inputs
