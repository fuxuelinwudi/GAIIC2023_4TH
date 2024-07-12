# coding:utf-8

import torch
import torch.nn.functional as F


def process_tgt(tgt_matrix, pad_token_id=0):
    contrastive_labels = tgt_matrix.clone()
    contrastive_labels[contrastive_labels[:, :] == pad_token_id] = 0
    contrastive_labels[contrastive_labels[:, :] != pad_token_id] = 1
    return contrastive_labels.type(torch.FloatTensor)


def label_smoothed_nll_loss(contrastive_scores, contrastive_labels, eps=0.0):
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.
    '''
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss = -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss) / contrastive_labels.sum()
    return loss
