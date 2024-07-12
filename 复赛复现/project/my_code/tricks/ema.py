# coding:utf-8


import torch

class EMA(object):

    def __init__(
            self,
            parameters,
            decay,
            use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        params = []
        for p in parameters:
            if p.requires_grad:
                params.append(p)
        for s_param, param in zip(self.shadow_params, params):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        params = []
        for p in parameters:
            if p.requires_grad:
                params.append(p)
        for c_param, param in zip(self.collected_params, params):
            if param.requires_grad:
                param.data.copy_(c_param.data)
