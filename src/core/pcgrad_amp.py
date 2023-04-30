# MIT License
#
# Copyright (c) 2022 Antoine Nzeyimana
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import random
import numpy as np
import torch

class PCGradAMP():
    def __init__(self, total_tasks, optimizer, scaler=None):
        self.main_task = total_tasks[0]
        self.aux_tasks = total_tasks[1:]
        self.total_tasks = total_tasks
        self.num_tasks = len(total_tasks)
        self._optim = optimizer
        self._scaler = scaler

        self.count          = {t:1 for t in self.aux_tasks} 
        self.count_005      = {t:1 for t in self.aux_tasks}
        self.count_01       = {t:1 for t in self.aux_tasks}
        self.total_count    = {t:1 for t in self.aux_tasks}
        self.percentage     = {t:1 for t in self.aux_tasks}
        self.percentage_005 = {t:1 for t in self.aux_tasks}
        self.percentage_01  = {t:1 for t in self.aux_tasks}
    
    def reset_count(self):
        self.count          = {t:1 for t in self.aux_tasks}
        self.count_005      = {t:1 for t in self.aux_tasks}
        self.count_01       = {t:1 for t in self.aux_tasks}
        self.total_count    = {t:1 for t in self.aux_tasks}

    def get_latest_ratio(self, types='0'):
        if types== '01':
            for t in self.aux_tasks:
                self.percentage_01[t] = round(self.count_01[t]/self.total_count[t], 2)
            return self.percentage_01
        elif types== '005':
            for t in self.aux_tasks:
                self.percentage_005[t] = round(self.count_005[t]/self.total_count[t], 2)
            return self.percentage_005
        else:
            for t in self.aux_tasks:
                self.percentage[t] = round(self.count[t]/self.total_count[t], 2)
            return self.percentage 
    
    def backward(self, mt_losses, gradient_scale, game_name_list, max_grad_norm):
        assert len(game_name_list) == len(mt_losses)

        main_grads = []
        aux_grads = []
        # backward
        for loss_id, loss in enumerate(mt_losses):
            self._optim.zero_grad()

            if self._scaler is not None:
                h = loss.register_hook(lambda grad: grad * gradient_scale)
                self._scaler.scale(loss).backward(retain_graph=True)
                h.remove()
                grad, shape, has_grad = self._retrieve_grad()
                current_scale = self._scaler.get_scale()
                grad = grad / current_scale
            else:
                h = loss.register_hook(lambda grad: grad * gradient_scale)
                loss.backward(retain_graph=True)
                h.remove()
                grad, shape, has_grad = self._retrieve_grad()
            
            torch.clamp(grad, -max_grad_norm, max_grad_norm, out=grad) 
            if loss_id == 0:
                main_grads.append((grad, shape, has_grad))
            else:
                aux_name = game_name_list[loss_id]
                aux_grads.append((aux_name, (grad, shape, has_grad)))

        self._optim.zero_grad()
        self._project_conflicting(main_grads, aux_grads)

    def _project_conflicting(self, main_grads, aux_grads):

        adapt_grad = main_grads[0][0]

        for idx, aux in enumerate(aux_grads):
            aux_name, aux_info = aux
            aux_grad = aux_info[0]
            cos_sim = torch.dot(adapt_grad, aux_grad) / (torch.norm(adapt_grad) * torch.norm(aux_grad))
            cos_sim = cos_sim.item()
            self.total_count[aux_name] += 1
            if cos_sim > 0:
                self.count[aux_name] += 1
            if cos_sim > 0.05:
                self.count_005[aux_name] += 1
            if cos_sim > 0.1:
                self.count_01[aux_name] += 1
            #print(cos_sim, aux_name, self.count[aux_name]/self.total_count[aux_name], self.total_count[aux_name])

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []

        groups = self._optim.param_groups
        for idx, group in enumerate(groups):
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p, dtype=torch.int8).to(p.device))
                else:
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p, dtype=torch.int8).to(p.device))

        grad_flatten = self._flatten_grad(grad, shape)
        has_grad_flatten = self._flatten_grad(has_grad, shape)
        return grad_flatten, shape, has_grad_flatten
