# coding:utf-8

import torch


class FGM:
    def __init__(self, model, eps=0.5):
        self.model = model
        self.backup = {}
        self.emb_name = 'embeddings.word_embeddings.'  # shared.weight, embeddings.word_embeddings.
        self.epsilon = eps

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model, eps=0.5):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = eps
        self.emb_name = 'embeddings.word_embeddings.'  # shared.weight, embeddings.word_embeddings.
        self.alpha = 0.3

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class AWP:
    """
    Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook

    for relation extration

    link : https://zhuanlan.zhihu.com/p/563641649
    """

    def __init__(self, model, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


# ===================
# freelb


def getDelta(
        attention_mask, embeds_init,
        adv_init_mag, adv_norm_type='l2'
):
    delta = None
    batch = embeds_init.shape[0]
    length = embeds_init.shape[-2]
    dim = embeds_init.shape[-1]

    attention_mask = attention_mask.view(-1, length)
    embeds_init = embeds_init.view(-1, length, dim)
    if adv_init_mag > 0:  # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
        input_mask = attention_mask.to(embeds_init)
        input_lengths = torch.sum(input_mask, 1)
        if adv_norm_type == "l2":
            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = input_lengths * embeds_init.size(-1)
            mag = adv_init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
        elif adv_norm_type == "linf":
            delta = torch.zeros_like(embeds_init).uniform_(-adv_init_mag, adv_init_mag)
            delta = delta * input_mask.unsqueeze(2)
    else:
        delta = torch.zeros_like(embeds_init)  # 扰动初始化

    return delta.view(batch, -1, length, dim)


def updateDelta(
        delta, delta_grad, embeds_init,
        adv_lr, adv_max_norm
):
    batch = delta.shape[0]
    length = delta.shape[-2]
    dim = delta.shape[-1]
    delta = delta.view(-1, length, dim)
    delta_grad = delta_grad.view(-1, length, dim)

    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
    denorm = torch.clamp(denorm, min=1e-8)
    delta = (delta + adv_lr * delta_grad / denorm).detach()
    if adv_max_norm > 0:
        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
        exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)
        reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
        #        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        delta = (delta * reweights).detach()

    return delta.view(batch, -1, length, dim)




