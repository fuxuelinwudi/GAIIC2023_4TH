# coding:utf-8


import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None, reduce='mean'):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        pad_mask = ~pad_mask  # 取反， 把 pad 的地方置为 1，mask 掉
        pad_mask = pad_mask.unsqueeze(-1).repeat(1, 1, p.shape[-1])
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    if reduce == 'sum':
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
    else:
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
