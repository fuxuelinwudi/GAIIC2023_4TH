# coding:utf-8


import torch
import torch.nn as nn
import numpy as np


def get_seq_len(inputs):
    lengths = torch.sum(inputs['attention_mask'], dim=-1)
    return lengths.detach().cpu().numpy()


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(InfoNCE, self).__init__()
        self.lower_size = 300
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, self.lower_size),
                                    nn.ReLU(),
                                    nn.Linear(self.lower_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [s_size, s_size, 1]

        lower_bound = T0.mean() - (
                    T1.logsumexp(dim=1).mean() - np.log(sample_size))  # torch.log(T1.exp().mean(dim = 1)).mean()

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


class CLUBv2(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, lr=1e-3, beta=0):
        super(CLUBv2, self).__init__()
        self.hiddensize = y_dim
        self.version = 2
        self.beta = beta

    def mi_est_sample(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        positive = torch.zeros_like(y_samples)

        prediction_1 = y_samples.unsqueeze(1)  # [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean(), positive.sum(dim = -1).mean(), negative.sum(dim = -1).mean()

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, x_samples, y_samples, steps=None):
        # no performance improvement, not enabled
        if steps:
            beta = self.beta if steps > 1000 else self.beta * steps / 1000  # beta anealing
        else:
            beta = self.beta

        return self.mi_est_sample(x_samples, y_samples) * self.beta


def feature_ranking(grad, cl=0.5, ch=0.9):
    n = len(grad)
    import math
    lower = math.ceil(n * cl)
    upper = math.ceil(n * ch)
    norm = torch.norm(grad, dim=1)  # [seq_len]
    _, ind = torch.sort(norm)
    res = []
    for i in range(lower, upper):
        res += ind[i].item(),
    return res


def local_robust_feature_selection(inputs, grad, cl, ch):
    grads = []
    lengths = get_seq_len(inputs)
    lengths = [int(item) for item in lengths]
    for i, length in enumerate(lengths):
        grads.append(grad[i, :length])
    indices = []
    nonrobust_indices = []
    for i, grad in enumerate(grads):
        indices.append(feature_ranking(grad, cl, ch))
        nonrobust_indices.append([x for x in range(lengths[i]) if x not in indices])
    return indices, nonrobust_indices


def _get_local_robust_feature_regularizer(mi_estimator, hidden_states, local_robust_features,
                                          train_batch_size, seed):
    last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
    sentence_embeddings = last_hidden[:, 0]  # batch x 768
    local_embeddings = []
    global_embeddings = []
    for i, local_robust_feature in enumerate(local_robust_features):
        for local in local_robust_feature:
            local_embeddings.append(embedding_layer[i, local])
            global_embeddings.append(sentence_embeddings[i])

    lower_bounds = []
    from sklearn.utils import shuffle
    local_embeddings, global_embeddings = shuffle(local_embeddings, global_embeddings, random_state=seed)
    for i in range(0, len(local_embeddings), train_batch_size):
        local_batch = torch.stack(local_embeddings[i: i + train_batch_size])
        global_batch = torch.stack(global_embeddings[i: i + train_batch_size])
        lower_bounds += mi_estimator(local_batch, global_batch),
    return -torch.stack(lower_bounds).mean()


def _train_mi_estimator(mi_estimator, hidden_states, inputs=None):
    last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
    sentence_embedding = last_hidden[:, 0]  # batch x 768
    if mi_estimator.version == 0:
        embedding_layer = torch.reshape(embedding_layer, [embedding_layer.shape[0], -1])
        return mi_estimator.update(embedding_layer, sentence_embedding)
    elif mi_estimator.version == 1:
        return mi_estimator.update(embedding_layer, last_hidden)
    elif mi_estimator.version == 2:
        return mi_estimator.update(embedding_layer, sentence_embedding)
    elif mi_estimator.version == 3:
        embeddings = []
        lengths = get_seq_len(inputs)
        lengths = [int(item) for item in lengths]
        for i, length in enumerate(lengths):
            embeddings.append(embedding_layer[i, :length])
        embeddings = torch.cat(embeddings)  # [-1, 768]
        return mi_estimator.update(embedding_layer, embeddings)


def _train_mi_upper_estimator(mi_upper_estimator, hidden_states, inputs=None):
    last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
    sentence_embedding = last_hidden[:, 0]  # batch x 768
    if mi_upper_estimator.version == 0:
        embedding_layer = torch.reshape(embedding_layer, [embedding_layer.shape[0], -1])
        return mi_upper_estimator.update(embedding_layer, sentence_embedding)
    elif mi_upper_estimator.version == 1:
        return mi_upper_estimator.update(embedding_layer, last_hidden)
    elif mi_upper_estimator.version == 2:
        return mi_upper_estimator.update(embedding_layer, sentence_embedding)
    elif mi_upper_estimator.version == 3:
        embeddings = []
        lengths = get_seq_len(inputs)
        lengths = [int(item) for item in lengths]
        for i, length in enumerate(lengths):
            embeddings.append(embedding_layer[i, :length, :])
        embeddings = torch.cat(embeddings)  # [-1, 768]
        return mi_upper_estimator.update(embedding_layer, embeddings)
