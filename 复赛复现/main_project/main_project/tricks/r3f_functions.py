# coding:utf-8


import torch
import torch.nn.functional as F


def _get_symm_kl(noised_logits, input_logits):
    return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
        F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
        F.softmax(noised_logits, dim=-1, dtype=torch.float32),
        reduction="sum",
    )
    ) / noised_logits.size(0)

