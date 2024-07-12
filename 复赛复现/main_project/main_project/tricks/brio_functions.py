# coding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from configurations.train_config_cpt import Config


user_args = Config()


def ranking_loss(score, summary_score=None, margin=0, gold_margin=0,
                 gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

    if no_gold:
        return TotalLoss

    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


def compute_brio_loss(
        output,
        candidate_id,
        cand_mask,
        labels,
        label_pad_token_id=-100,
        normalize=True,
        score_mode='base',
        adding=0,
        scale=1.0,
        mle_weight=1.0,
        rank_weight=100.0,
        vocab_size=1415
):

    if user_args.lbs:
        lbs_mle_fn = label_smoothing_loss(ignore_index=label_pad_token_id, epsilon=user_args.smoothing)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=label_pad_token_id)

    # [bz * cand_num, seq_len, hidden_dim]
    output = output.logits

    # [bz, cand_num, seq_len, hidden_dim]
    output = output.view(user_args.batch_size, -1, output.size(1), output.size(2))

    # real ans probs, [bsz, seq_len, hidden_dim]
    probs = output[:, 0, :, :]

    # [bsz, cand_num + 1, seq_len, 1]
    candidate_id = candidate_id.unsqueeze(-1)

    if normalize:
        if score_mode == "log":
            _output = F.log_softmax(output, dim=3)
        else:
            _output = F.softmax(output, dim=3)
        scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
    else:
        scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]

    cand_mask = cand_mask.float()
    scores = torch.mul(scores, cand_mask).sum(-1) / (
            (cand_mask.sum(-1) + adding) ** user_args.length_penalty)  # [bz, cand_num]

    score = scores[:, 1:]
    summary_score = scores[:, 0]

    similarity, gold_similarity = score, summary_score

    similarity = similarity * scale
    gold_similarity = gold_similarity * scale

    rank_loss = ranking_loss(similarity, gold_similarity, 0.001, 0, 0)

    if user_args.lbs:
        mle_loss = lbs_mle_fn(probs, labels)
    else:
        mle_loss = mle_fn(probs.reshape(-1, vocab_size), labels.view(-1))

    brio_loss = mle_weight * mle_loss + rank_weight * rank_loss

    return brio_loss, mle_loss.item(), rank_loss.item()