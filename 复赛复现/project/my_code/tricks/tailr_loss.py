import torch
import torch.nn as nn
import torch.nn.functional as F


class TaiLr(object):
    def __init__(
            self,
            label_smoothing=0.0,
            density_min_weight=0.0,
            density_ratio_threshold=0.8,
            report_accuracy=False,
    ):
        super().__init__()

        self.eps = label_smoothing
        self.report_accuracy = report_accuracy
        self.min_weight = density_min_weight
        self.gamma = density_ratio_threshold

    def get_normalized_probs(
        self,
        net_output,
        log_probs=True,
    ):

        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)

        return F.softmax(logits, dim=-1)

    def tailr_loss(self, lprobs, target, epsilon, min_weight, gamma, probs_model, ignore_index=None, reduce=True):
        """
        args：
            lprobs：取log的logit，[bs*seq_len,dim]
            target: [bs*seq_len]
            probs_model: [bs*seq_len,dim]
        计算公式为：设p为模型预测的概率, r为 gamma
        loss = -[p/(r+(1-r)*p)]*log(p)
        """

        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # logp

        # -100 to 0
        mask = (target < 0).float()
        # 将符合条件的元素的值置为0
        target = (target * (1 - mask) + mask * 0).to(torch.int64)

        nll_loss = -lprobs.gather(dim=-1, index=target)
        # 额外的一个 smooth loss，不是 tailr 官方里面的
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        weight_theta_hat = probs_model.gather(dim=-1, index=target)

        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)

        with torch.no_grad():
            # p/[r+(1-r)p]
            xx = torch.log(weight_theta_hat)
            yy = torch.log((gamma + (1 - gamma) * weight_theta_hat))
            zz = torch.exp(xx-yy)
            weight_theta_hat = torch.exp((torch.log(weight_theta_hat) - torch.log((gamma + (1 - gamma) * weight_theta_hat))))
            weight_theta_hat = torch.clamp(weight_theta_hat, min=min_weight, max=1.0)

        tailr_loss = weight_theta_hat * nll_loss
        # Can also adjust smooth loss accordingly; but no big impact
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            tailr_loss = tailr_loss.sum()
        eps_i = epsilon / (lprobs.size(-1))
        loss = (1. - epsilon) * tailr_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_official_loss(self, net_output, target, reduce=True):
        """官方实现方式"""
        lprobs = self.get_normalized_probs(net_output, log_probs=True)

        probs_model = self.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1))
        loss, nll_loss = self.tailr_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            self.eps,
            ignore_index=-100,
            reduce=reduce,
            min_weight=self.min_weight,
            gamma=self.gamma,
            probs_model=probs_model,
        )
        return loss, nll_loss, lprobs

#
# batch_size = 4
# num_classes = 10
# seq_len = 10
# output = torch.randn(batch_size, seq_len, num_classes)
# output = nn.LogSoftmax(dim=1)(output)
# targets = torch.LongTensor(batch_size, seq_len).random_(num_classes)
# print(output.shape, targets.shape)
#
# tailr_loss = TaiLr()
# print(output.shape, targets.shape)
# loss, nll_loss, lprobs = tailr_loss.compute_official_loss(output, targets)
# print(loss.item())
