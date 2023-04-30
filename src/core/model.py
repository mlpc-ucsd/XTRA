import typing
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


# from core.game import Action
class Action(object):
    pass


class NetworkOutput(typing.NamedTuple):
    value: float
    reward_sum: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]
    reward_hidden: object


def concat_output_value(output_lst):
    # for numpy
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst

def concat_output(output_lst):
    # for numpy
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_c_lst, reward_hidden_h_lst =[], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.reward_sum)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_c_lst.append(output.reward_hidden[0].squeeze(0))
        reward_hidden_h_lst.append(output.reward_hidden[1].squeeze(0))
    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    # hidden_state_lst = torch.cat(hidden_state_lst, 0)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_c_lst = np.expand_dims(np.concatenate(reward_hidden_c_lst), axis=0)
    reward_hidden_h_lst = np.expand_dims(np.concatenate(reward_hidden_h_lst), axis=0)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (reward_hidden_c_lst, reward_hidden_h_lst)


class BaseMuZeroNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform, lstm_hidden_size):
        super(BaseMuZeroNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform
        self.lstm_hidden_size = lstm_hidden_size

    def rotate(self, images, angle):
        k = angle // 90
        # images: (bz, num_stacked * 3, h, w)
        images = torch.rot90(images, k=k, dims=[2, 3])
        return images
        
    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def initial_inference(self, obs, actions_id=None, keep_tensor=False) :
        num = obs.size(0)

        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

        if actions_id is not None and len(actions_id) < actor_logit.shape[1]:
            actor_logit = actor_logit[:, actions_id]


        if not self.training:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            

            if keep_tensor:
                reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to('cuda'), torch.zeros(1, num, self.lstm_hidden_size).to('cuda'))

            else:
                state = state.detach().cpu().numpy()
                actor_logit = actor_logit.detach().cpu().numpy()
                # zero initialization for reward (value prefix) hidden states
                reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy(),
                                torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy())
        else:
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to('cuda'), torch.zeros(1, num, self.lstm_hidden_size).to('cuda'))

        return NetworkOutput(value, [0. for _ in range(num)], actor_logit, state, reward_hidden)

    def recurrent_inference(self, hidden_state, reward_hidden, action, actions_id=None, overwrite=False):
        state, reward_hidden, reward_sum = self.dynamics(hidden_state, reward_hidden, action)
        actor_logit, value = self.prediction(state)

        if actions_id is not None and len(actions_id) < actor_logit.shape[1]:
            actor_logit = actor_logit[:, actions_id]
        #TODO: what is overwrite?
        if not self.training and not overwrite:
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            reward_sum = self.inverse_reward_transform(reward_sum).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            reward_hidden = (reward_hidden[0].detach().cpu().numpy(), reward_hidden[1].detach().cpu().numpy())
            actor_logit = actor_logit.detach().cpu().numpy()

        return NetworkOutput(value, reward_sum, actor_logit, state, reward_hidden)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)