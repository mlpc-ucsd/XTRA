# modify game name, aux head

from telnetlib import Telnet
import logging
from multiprocessing.dummy import freeze_support
from contextlib import nullcontext
import psutil
import os
import ray
import math
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from ray.util.queue import Queue
from torch.utils.tensorboard import SummaryWriter
from ray.util.multiprocessing import Pool
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import core.ctree.cytree as cytree
from core.model import NetworkOutput
from .mcts import MCTS, get_node_distribution
from .replay_buffer import ReplayBuffer
from .test import test
from .utils import select_action, profile, prepare_observation_lst, LinearSchedule
from .game import GameHistory, prepare_multi_target, prepare_multi_target_only_value
from .game_detach import prepare_multi_target_cpu, prepare_multi_target_only_value_cpu, MCTS_Storage, PrepareTargetGpuActor, prepare_imitation_learning, imitation_postprocess
import time
import numpy as np
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP_apex
except:
    pass
import copy
from core.pcgrad_amp import PCGradAMP

train_logger = logging.getLogger('train')
plan_logger = logging.getLogger('plan')
test_logger = logging.getLogger('train_test')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def _log(config, step_count, log_data, replay_buffer, lr, shared_storage, game_name=None):
    total_loss, weighted_loss, loss, reg_loss, policy_loss, reward_sum_loss, value_loss, consistency_loss = log_data

    replay_episodes_collected, replay_buffer_size, priorities, total_num, worker_logs = ray.get([
        replay_buffer.episodes_collected.remote(), replay_buffer.size.remote(),
        replay_buffer.get_priorities.remote(), replay_buffer.get_total_len.remote(),
        shared_storage.get_worker_logs.remote()])

    worker_ori_reward, worker_reward, worker_reward_max, worker_eps_len, worker_eps_len_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions = worker_logs

    _msg = '#{:<10} Total Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Sum Loss: {:<8.3f} Consistency Loss: {:<8.3f} ] ' \
           'Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Transition Number: {:<8.3f}k ' \
           'Batch Size: {:<10d} Lr: {:<8.6f} Game_Name {:<10s}'
    _msg = _msg.format(step_count, total_loss, weighted_loss, policy_loss, value_loss, reward_sum_loss, consistency_loss,
                       replay_episodes_collected, replay_buffer_size, total_num / 1000, config.batch_size, lr, game_name)
    train_logger.info(_msg)

    if test_dict is not None:
        mean_score = np.mean(test_dict['mean_score'])
        std_score = np.mean(test_dict['std_score'])
        test_msg = '#{:<10} Test Mean Score: [{:<10}]({:<10})'.format(test_counter, mean_score, std_score)
        test_logger.info(test_msg)


class BatchStorage(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.batch_queue = Queue(maxsize=size)

    def push(self, batch):
        if self.batch_queue.qsize() <= self.threshold:
            self.batch_queue.put(batch)

    def pop(self):
        if self.batch_queue.qsize() > 0:
            return self.batch_queue.get()
        else:
            return None

    def get_len(self):
        return self.batch_queue.qsize()

@ray.remote
class SharedStorage(object):
    def __init__(self, model, target_model=None, latest_model=None, planformer=None):
        self.step_counter = 0
        self.test_counter = 0
        self.model = model
        self.target_model = target_model
        self.latest_model = latest_model
        self.ori_reward_log = []
        self.reward_log = []
        self.reward_max_log = []
        self.test_dict_log = {}
        self.eps_lengths = []
        self.eps_lengths_max = []
        self.temperature_log = []
        self.visit_entropies_log = []
        self.priority_self_play_log = []
        self.distributions_log = {
            'depth': [],
            'visit': []
        }
        self.start = False

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def get_target_weights(self):
        return self.target_model.get_weights()

    def set_target_weights(self, weights):
        return self.target_model.set_weights(weights)

    def get_latest_weights(self):
        return self.latest_model.get_weights()

    def set_latest_weights(self, weights):
        return self.latest_model.set_weights(weights)

    def get_planformer_weights(self):
        return self.planformer.get_weights()

    def set_planformer_weights(self, weights):
        return self.planformer.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_len_max, eps_ori_reward, eps_reward, eps_reward_max, temperature, visit_entropy, priority_self_play, distributions):
        self.eps_lengths.append(eps_len)
        self.eps_lengths_max.append(eps_len_max)
        self.ori_reward_log.append(eps_ori_reward)
        self.reward_log.append(eps_reward)
        self.reward_max_log.append(eps_reward_max)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)
        self.priority_self_play_log.append(priority_self_play)

        for key, val in distributions.items():
            self.distributions_log[key] += val

    def add_test_log(self, test_counter, test_dict):
        self.test_counter = test_counter
        for key, val in test_dict.items():
            if key not in self.test_dict_log.keys():
                self.test_dict_log[key] = []
            self.test_dict_log[key].append(val)

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            ori_reward = sum(self.ori_reward_log) / len(self.ori_reward_log)
            reward = sum(self.reward_log) / len(self.reward_log)
            reward_max = sum(self.reward_max_log) / len(self.reward_max_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            eps_lengths_max = sum(self.eps_lengths_max) / len(self.eps_lengths_max)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)
            priority_self_play = sum(self.priority_self_play_log) / len(self.priority_self_play_log)
            distributions = self.distributions_log

            self.ori_reward_log = []
            self.reward_log = []
            self.reward_max_log = []
            self.eps_lengths = []
            self.eps_lengths_max = []
            self.temperature_log = []
            self.visit_entropies_log = []
            self.priority_self_play_log = []
            self.distributions_log = {
                'depth': [],
                'visit': []
            }

        else:
            ori_reward = None
            reward = None
            reward_max = None
            eps_lengths = None
            eps_lengths_max = None
            temperature = None
            visit_entropy = None
            priority_self_play = None
            distributions = None

        if len(self.test_dict_log) > 0:
            test_dict = self.test_dict_log

            self.test_dict_log = {}
            test_counter = self.test_counter
        else:
            test_dict = None
            test_counter = None

        return ori_reward, reward, reward_max, eps_lengths, eps_lengths_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions


def update_weights(model, batch_list, optimizer, replay_buffer, config, scaler, game_info=None, replay_buffer_dict=None, multi_game=False, bz_list=None, step_count=None, pcgrad=None):
    assert len(batch_list) >=1

    # zero gradients 
    optimizer.zero_grad()

    loss_log = []
    gamename_log = []
    
    assert multi_game == True, "online finetuning mixed with pretrained tasks"
    obs_batch_ori_list, action_batch_list, mask_batch_list, target_reward_sum_list, target_value_list, target_policy_list, indices_list, weights_lst_list, make_time_list, game_name_list = zip(*batch_list)

    obs_batch_ori = np.concatenate(obs_batch_ori_list, axis=0)
    action_batch = np.concatenate(action_batch_list, axis=0)
    mask_batch = np.concatenate(mask_batch_list, axis=0)
    target_reward_sum = np.concatenate(target_reward_sum_list, axis=0)
    target_value = np.concatenate(target_value_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)
    weights_lst = np.concatenate(weights_lst_list, axis=0)
    make_time = np.concatenate(make_time_list, axis=0)

    actions_id_list = [ game_info[game_name.item()]['action_id'] for game_name in game_name_list]

    target_policy_full_list = []
    for target_id in range(len(target_policy_list)):
        target_action_id = actions_id_list[target_id]
        target_policy_current = target_policy_list[target_id]
        target_policy_current_full = np.zeros((target_policy_current.shape[0], 6, 18))
        target_policy_current_full[:, :, target_action_id] = target_policy_current
        target_policy_full_list.append(target_policy_current_full)
    target_policy = np.concatenate(target_policy_full_list, axis=0)
    #ratio = int(target_policy_list[0].shape[0] / target_policy_list[-1].shape[0])
    gamename_log = [game_name.item()[:-len('NoFrameskip-v4')] for game_name in game_name_list]

    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float() / 255.0
    obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
    obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]

    if config.use_augmentation:  
        obs_batch = config.transform(obs_batch)
        obs_target_batch = config.transform(obs_target_batch)

    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_reward_sum = torch.from_numpy(target_reward_sum).to(config.device).float()
    target_value = torch.from_numpy(target_value).to(config.device).float()
    target_policy = torch.from_numpy(target_policy).to(config.device).float()
    weights = torch.from_numpy(weights_lst).to(config.device).float()

    batch_size = obs_batch.size(0)
    assert batch_size == target_reward_sum.size(0)

    transformed_target_reward_sum = config.scalar_transform(target_reward_sum)
    target_reward_sum_phi = config.reward_phi(transformed_target_reward_sum)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)

    with autocast() if config.amp_type == 'torch_amp' else nullcontext():
        value, _, policy_logits, hidden_state, reward_hidden = model.initial_inference(obs_batch)

    scaled_value = config.inverse_value_transform(value)

    # Reference: Appendix G
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps
    reward_priority = []

    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    reward_sum_loss = torch.zeros(batch_size, device=config.device)
    consistency_loss = torch.zeros(batch_size, device=config.device)

    gradient_scale = 1 / config.num_unroll_steps

    with autocast() if config.amp_type == 'torch_amp' else nullcontext():
        for step_i in range(config.num_unroll_steps):
            value, reward_sum, policy_logits, hidden_state, reward_hidden = model.recurrent_inference(hidden_state, reward_hidden, action_batch[:, step_i])#, actions_id_list)#, game_name)

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)

            if config.consistency_coeff > 0:
                _, _, _, presentation_state, _ = model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])#, actions_id_list)#, game_name=game_name)
                dynamic_proj = model.project(hidden_state, with_grad=True)#, game_name=game_name)
                observation_proj = model.project(presentation_state, with_grad=False)#, game_name=game_name)
                temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
            reward_sum_loss += config.scalar_reward_loss(reward_sum, target_reward_sum_phi[:, step_i])
            hidden_state.register_hook(lambda grad: grad * 0.5)

            scaled_reward_sums = config.inverse_reward_transform(reward_sum.detach())

            l1_prior = torch.nn.L1Loss(reduction='none')(scaled_reward_sums.squeeze(-1), target_reward_sum[:, step_i])
            reward_priority.append(l1_prior.detach().cpu().numpy())


    game_name_list = [i.tolist() for i in game_name_list]
    aux_list = game_name_list[1:]
    if config.optimizer == 'pcgrad':
        if step_count == 0:
            current_ratio = [1.0] * len(aux_list)
            with open(config.exp_path + '/current_ratio.txt', 'w') as f:
                f.write(str(step_count) + str(current_ratio) + str(current_ratio) + str(current_ratio)+ '\n')
        #TODO: add variables as arguments
        grad_steps = 500
        normal_steps  = 5000
        write_stats=False
        if step_count % normal_steps < grad_steps and step_count >= 10000:
            start_pcgrad = True
        else:
            # if it is first normal steps, then write stats
            if step_count % normal_steps == grad_steps:
                write_stats=True
            start_pcgrad = False
        
        # This is not for actual update
        with autocast() if config.amp_type == 'torch_amp' else nullcontext():
            consistency_loss_adapt = consistency_loss[:batch_size // 2]
            policy_loss_adapt = policy_loss[:batch_size // 2]
            value_loss_adapt = value_loss[:batch_size // 2]
            reward_sum_loss_adapt = reward_sum_loss[:batch_size // 2]
            weights_adapt = weights[:batch_size // 2]

            loss_adapt = (config.value_loss_coeff * value_loss_adapt + config.policy_loss_coeff * policy_loss_adapt + 
                        config.reward_loss_coeff * reward_sum_loss_adapt + config.consistency_coeff * consistency_loss_adapt)
            weighted_loss_adapt = (weights_adapt * loss_adapt).mean()

            # split losses to 1/2 batch
            # THis MUST BE EVEN BATCHES
            bz_aux = batch_size // 2
            bz_aux_game = bz_aux // (len(aux_list))

            consistency_loss_aux = consistency_loss[batch_size // 2:]
            policy_loss_aux = policy_loss[batch_size // 2:]
            value_loss_aux = value_loss[batch_size // 2:]
            reward_sum_loss_aux = reward_sum_loss[batch_size // 2:]
            weight_aux = weights[batch_size // 2:]

            loss_aux = []
            for aux_i in range(len(aux_list)):
                loss_aux_i = (config.value_loss_coeff * value_loss_aux[aux_i*bz_aux_game:(aux_i+1)*bz_aux_game] + 
                            config.policy_loss_coeff * policy_loss_aux[aux_i*bz_aux_game:(aux_i+1)*bz_aux_game] +
                            config.reward_loss_coeff * reward_sum_loss_aux[aux_i*bz_aux_game:(aux_i+1)*bz_aux_game] + 
                            config.consistency_coeff * consistency_loss_aux[aux_i*bz_aux_game:(aux_i+1)*bz_aux_game])
                weighted_loss_aux_i = (weight_aux[aux_i*bz_aux_game:(aux_i+1)*bz_aux_game] * loss_aux_i).mean()
                loss_aux.append(weighted_loss_aux_i)

            loss_list = [weighted_loss_adapt]+loss_aux
        
        if start_pcgrad:
            pcgrad.backward(loss_list, gradient_scale, game_name_list, config.max_grad_norm)

        if write_stats:
            # update pcgrad @ e.g. 500, 5500
            pcgrad.get_latest_ratio(types='0')
            pcgrad.get_latest_ratio(types='005')
            pcgrad.get_latest_ratio(types='01')
            with open(config.exp_path + '/current_ratio.txt', 'a') as f:
                f.write(str(step_count) + str(pcgrad.percentage) + str(pcgrad.percentage_005) +str(pcgrad.percentage_01)+'\n')
            
            pcgrad.reset_count()

        # get the ratio of the current batch
        current_ratio = pcgrad.percentage_01
        print(current_ratio) 

    # L2 reg
    parameters = model.parameters()
    optimizer.zero_grad() 

# organize marker

    # update the model
    if config.optimizer == 'pcgrad':
        # loss
        if config.aux_data_decay:
            # select tasks with ratio (cosine similarity)
            with autocast() if config.amp_type == 'torch_amp' else nullcontext():
                for aux_i in range(len(aux_list)):
                    start_idx = aux_i*bz_aux_game
                    end_idx = (aux_i+1)*bz_aux_game

                    consistency_loss_adapt = torch.cat((consistency_loss_adapt, consistency_loss_aux[start_idx:end_idx]), dim=0)
                    policy_loss_adapt = torch.cat((policy_loss_adapt, policy_loss_aux[start_idx:end_idx]), dim=0)
                    value_loss_adapt = torch.cat((value_loss_adapt, value_loss_aux[start_idx:end_idx]), dim=0)
                    reward_sum_loss_adapt = torch.cat((reward_sum_loss_adapt, reward_sum_loss_aux[start_idx:end_idx]), dim=0)
                    
                    # TODO:reduce the weight of the aux data
                    c_ratio = current_ratio[aux_list[aux_i]] 
                    weights_adapt = torch.cat((weights_adapt, weight_aux[start_idx:end_idx]*c_ratio), dim=0)
                    
            loss = (config.consistency_coeff * consistency_loss_adapt + config.policy_loss_coeff * policy_loss_adapt +
                    config.value_loss_coeff * value_loss_adapt + config.reward_loss_coeff * reward_sum_loss_adapt)
            weighted_loss = (weights_adapt * loss).mean()               
        else:
            # no ratio. load all the data
            loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss +
                    config.value_loss_coeff * value_loss + config.reward_loss_coeff * reward_sum_loss)
            weighted_loss = (weights * loss).mean()

        # backward
        if config.amp_type == 'torch_amp':
            with autocast():
                total_loss = weighted_loss
                total_loss.register_hook(lambda grad: grad * gradient_scale)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()

        # log
        if config.aux_data_decay:
            loss_data = (weighted_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss_adapt.mean().item(),
                reward_sum_loss_adapt.mean().item(), value_loss_adapt.mean().item(), consistency_loss_adapt.mean().item())
        else:
            loss_data = (weighted_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                reward_sum_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean().item())

        loss_log.append(loss_data)  

    else: 
        # Loss for the whole batch
        loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss +
                config.value_loss_coeff * value_loss + config.reward_loss_coeff * reward_sum_loss)
        weighted_loss = (weights * loss).mean()

        with autocast() if config.amp_type == 'torch_amp' else nullcontext():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)

        # backward
        if config.amp_type == 'none':
            total_loss.backward()
        elif config.amp_type == 'torch_amp':
            scaler.scale(total_loss).backward()

        if config.amp_type == 'torch_amp':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()

        # packing data for logging
        loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                    reward_sum_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean().item())
        loss_log.append(loss_data)  

    # update priorities
    if config.use_priority:
        reward_priority = np.mean(reward_priority, 0)
        new_priority = (1 - config.priority_reward_ratio) * value_priority + config.priority_reward_ratio * reward_priority

        num_games = len(game_name_list)
        acc_bz = 0 
        for gdx in range(num_games):
            bz = bz_list[gdx]
            g_indices = indices[acc_bz:acc_bz+bz]
            g_new_priority = new_priority[acc_bz:acc_bz+bz]
            g_make_time = make_time[acc_bz:acc_bz+bz]
            replay_buffer_dict[gamename_log[gdx]+'NoFrameskip-v4'].update_priorities.remote(g_indices, g_new_priority, g_make_time)
            acc_bz += bz

    loss_log = np.mean(loss_log, 0).tolist()
    gamename_log = '+'.join(gamename_log)
    return loss_log, scaler, gamename_log

def consist_loss_func(f1, f2):
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count, scheduler, opt_default_lr, lr_min=None):
    # warmup
    if step_count < config.lr_warm_step:
        for idx in range(len(opt_default_lr)):
            if lr_min is not None: 
                lr = lr_min + (opt_default_lr[idx]-lr_min) * step_count / config.lr_warm_step
            else:
                lr = opt_default_lr[idx] * step_count / config.lr_warm_step
            param_group = optimizer.param_groups[idx]
            param_group['lr'] = lr
    else:
        if config.lr_type == 'cosine':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
        else:
            for idx in range(len(opt_default_lr)):
                lr = opt_default_lr[idx] * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
                
                param_group = optimizer.param_groups[idx]
                param_group['lr'] = lr

    return lr


def add_batch(batch, m_batch):
    # obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices_lst, weights_lst, make_time
    for i, m_bt in enumerate(m_batch):
        batch[i].append(m_bt)



@ray.remote(num_gpus=0.125)
class DataWorker(object):
    def __init__(self, rank, config, shared_storage, replay_buffer, game_info, multi_game=False):
        self.rank = rank
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.trajectory_pool = []
        self.pool_size = 1
        self.device = 'cuda'
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1
        self.game_info = game_info
        self.multi_game = multi_game
        print(f'multi_game: {self.multi_game}')

    def put(self, data):
        self.trajectory_pool.append(data)

    def put_last_trajectory(self, i, last_game_histories, last_game_priorities, game_histories):
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        last_game_histories[i].game_over()

        self.put((last_game_histories[i], last_game_priorities[i]))
        self.free()

        # reset last block
        last_game_histories[i] = None
        last_game_priorities[i] = None

    def len_pool(self):
        return len(self.trajectory_pool)

    def free(self):
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool, self.gap_step)
            del self.trajectory_pool[:]

    def get_priorities(self, i, pred_values_lst, search_values_lst):

        if self.config.use_priority and not self.config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            priorities = None

        return priorities

    def run_multi(self):

        game_name = self.config.env_name
        actions_num = self.game_info[game_name]['total_action']
        actions_id  = self.game_info[game_name]['action_id']
        
        # number of parallel mcts
        env_nums = self.config.p_mcts_num

        model = self.config.get_uniform_network()
        model.to(self.device)
        model.eval()


        start_training = False
        envs = [self.config.new_game(self.config.seed + self.rank * i) for i in range(env_nums)]

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(actions_num)
        # 100k benchmark
        total_transitions = 0
        max_transitions = self.config.total_transitions // self.config.num_actors
        with torch.no_grad():
            while True:
                trained_steps = ray.get(self.shared_storage.get_counter.remote())
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    break

                init_obses = [env.reset() for env in envs]
                dones = np.array([False for _ in range(env_nums)])
                game_histories = [GameHistory(envs[_].env.action_space, max_length=self.config.history_length,
                                              config=self.config) for _ in range(env_nums)]
                last_game_histories = [None for _ in range(env_nums)]
                last_game_priorities = [None for _ in range(env_nums)]

                # stack observation windows in boundary: s198, s199, s200, current s1 -> for not init trajectory
                stack_obs_windows = [[] for _ in range(env_nums)]

                for i in range(env_nums):
                    stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
                    game_histories[i].init(stack_obs_windows[i])

                # this the root value of MCTS
                search_values_lst = [[] for _ in range(env_nums)]
                # predicted value of target network
                pred_values_lst = [[] for _ in range(env_nums)]
                # observed n-step return

                eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums)
                step_counter = 0

                self_play_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []
                depth_distribution = []
                visit_count_distribution = []

                while not dones.all() and (step_counter <= self.config.max_moves):
                    if not start_training:
                        start_training = ray.get(self.shared_storage.get_start_signal.remote())

                    # get model
                    trained_steps = ray.get(self.shared_storage.get_counter.remote())
                    if trained_steps >= self.config.training_steps + self.config.last_steps:
                        # training is finished
                        return
                    if start_training and (total_transitions / max_transitions) > (trained_steps / self.config.training_steps):
                        # self-play is faster or finished
                        time.sleep(1)
                        continue

                    _temperature = np.array(
                        [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps) for env in
                         envs])

                    new_model_index = trained_steps // self.config.checkpoint_interval
                    if new_model_index > self.last_model_index:
                        self.last_model_index = new_model_index
                        # update model
                        weights = ray.get(self.shared_storage.get_weights.remote())
                        model.set_weights(weights)
                        model.to(self.device)
                        model.eval()

                        # log
                        if env_nums > 1:
                            if len(self_play_visit_entropy) > 0:
                                visit_entropies = np.array(self_play_visit_entropy).mean()
                                visit_entropies /= max_visit_entropy
                            else:
                                visit_entropies = 0.

                            if self_play_episodes > 0:
                                log_self_play_moves = self_play_moves / self_play_episodes
                                log_self_play_rewards = self_play_rewards / self_play_episodes
                                log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                            else:
                                log_self_play_moves = 0
                                log_self_play_rewards = 0
                                log_self_play_ori_rewards = 0

                            # depth_distribution = np.array(depth_distribution)
                            # visit_count_distribution = np.array(visit_count_distribution)
                            self.shared_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                            log_self_play_ori_rewards, log_self_play_rewards,
                                                                            self_play_rewards_max, _temperature.mean(),
                                                                            visit_entropies, 0,
                                                                            {'depth': depth_distribution,
                                                                             'visit': visit_count_distribution})
                            self_play_rewards_max = - np.inf

                    step_counter += 1
                    ## reset env if finished
                    for i in range(env_nums):
                        if dones[i]:

                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # store current block trajectory
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], priorities))
                            self.free()

                            envs[i].close()
                            init_obs = envs[i].reset()
                            game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            last_game_histories[i] = None
                            last_game_priorities[i] = None
                            stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                            game_histories[i].init(stack_obs_windows[i])

                            self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_rewards += eps_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1

                            pred_values_lst[i] = []
                            search_values_lst[i] = []
                            # end_tags[i] = False
                            eps_steps_lst[i] = 0
                            eps_reward_lst[i] = 0
                            eps_ori_reward_lst[i] = 0
                            visit_entropies_lst[i] = 0

                    stack_obs = [game_history.step_obs() for game_history in game_histories]
                    if self.config.image_based:
                        stack_obs = prepare_observation_lst(stack_obs)
                        stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                    else:
                        stack_obs = [game_history.step_obs() for game_history in game_histories]
                        stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)
                    
                    # planning with different mode
                    with autocast() if self.config.amp_type == 'torch_amp' else nullcontext():
                        network_output = model.initial_inference(stack_obs.float(), actions_id)
                    hidden_state_roots = network_output.hidden_state
                    reward_hidden_roots = network_output.reward_hidden
                    reward_sum_pool = network_output.reward_sum
                    policy_logits_pool = network_output.policy_logits.tolist()

                    roots = cytree.Roots(env_nums, actions_num, self.config.num_simulations)
                    noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * actions_num).astype(np.float32).tolist() for _ in range(env_nums)]
                    roots.prepare(self.config.root_exploration_fraction, noises, reward_sum_pool, policy_logits_pool)
                    
                    MCTS(self.config).run_multi(roots, model, hidden_state_roots, reward_hidden_roots, actions_id, multi_game=self.multi_game)

                    roots_distributions = roots.get_distributions()
                    roots_values = roots.get_values()
                    for i in range(env_nums):
                        deterministic = False
                        # if start_training, always use the predicted actions
                        # if resume model, always use the predicted actions
                        if start_training or self.config.model_path != '':
                            distributions, value, temperature, env = roots_distributions[i], roots_values[i], _temperature[i], envs[i]
                        # if not start_training, use equal probability to choose actions
                        else:
                            value, temperature, env = roots_values[i], _temperature[i], envs[i]
                            distributions = np.ones(actions_num)

                        action, visit_entropy = select_action(distributions, temperature=temperature, deterministic=deterministic)
                        obs, ori_reward, done, info = env.step(action)
                        if self.config.clip_reward:
                            clip_reward = np.sign(ori_reward)
                        else:
                            clip_reward = ori_reward

                        game_histories[i].store_search_stats(distributions, value)
                        game_histories[i].append(action, obs, clip_reward)

                        eps_reward_lst[i] += clip_reward
                        eps_ori_reward_lst[i] += ori_reward
                        dones[i] = done
                        visit_entropies_lst[i] += visit_entropy

                        eps_steps_lst[i] += 1
                        # if start_training:
                        total_transitions += 1

                        if self.config.use_priority and not self.config.use_max_priority and start_training:
                            pred_values_lst[i].append(network_output.value[i].item())
                            search_values_lst[i].append(roots_values[i])

                        # fresh stack windows
                        del stack_obs_windows[i][0]
                        stack_obs_windows[i].append(obs)

                        # if game history is full
                        if game_histories[i].is_full():
                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # calculate priority
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                            # save block trajectory
                            last_game_histories[i] = game_histories[i]
                            last_game_priorities[i] = priorities

                            # new block trajectory
                            game_histories[i] = GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            game_histories[i].init(stack_obs_windows[i])

                for i in range(env_nums):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        # pad over last block trajectory
                        if last_game_histories[i] is not None:
                            self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                        # store current block trajectory
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], priorities))
                        self.free()

                        self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                        self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                        self_play_rewards += eps_reward_lst[i]
                        self_play_ori_rewards += eps_ori_reward_lst[i]
                        self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                        self_play_moves += eps_steps_lst[i]
                        self_play_episodes += 1
                    else:
                        # not save this data
                        total_transitions -= len(game_histories[i])

                visit_entropies = np.array(self_play_visit_entropy).mean()
                visit_entropies /= max_visit_entropy

                if self_play_episodes > 0:
                    log_self_play_moves = self_play_moves / self_play_episodes
                    log_self_play_rewards = self_play_rewards / self_play_episodes
                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                else:
                    log_self_play_moves = 0
                    log_self_play_rewards = 0
                    log_self_play_ori_rewards = 0

                # depth_distribution = np.array(depth_distribution)
                # visit_count_distribution = np.array(visit_count_distribution)
                self.shared_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                log_self_play_ori_rewards, log_self_play_rewards,
                                                                self_play_rewards_max, _temperature.mean(),
                                                                visit_entropies, 0,
                                                                {'depth': depth_distribution,
                                                                 'visit': visit_count_distribution})

@ray.remote 
class BatchWorker_CPU(object):
    def __init__(self, worker_id, replay_buffer, storage, mcts_storage, config, game_info=None, multi_game=False, start_now=False):
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.mcts_storage = mcts_storage
        self.config = config

        self.last_model_index = -1
        self.batch_max_num = 20
        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps, initial_p=config.priority_prob_beta, final_p=1.0)

        self.game_info = game_info
        self.multi_game = multi_game
        self.count_idx = 0
        self.start_now = start_now

        self.aux_total_bz = int(self.config.batch_size//len(self.config.aux_data_list.split('/')))
        self.aux_schedule = None
    def run(self):
        start = False
        if self.start_now:
            start = True

        while True:
            # wait for starting
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(1)
                continue

            # TODO: use latest weights for policy reanalyze
            ray_data_lst = [self.storage.get_counter.remote(), self.storage.get_target_weights.remote()]
            trained_steps, target_weights = ray.get(ray_data_lst)



            if self.start_now: # aux worker
                #bz = int(self.config.batch_size//len(self.config.aux_data_list.split('/')))
                if self.aux_schedule: 
                    bz = int(self.aux_schedule.value(trained_steps))//2*2
                    # clip bz min to 1
                    bz = max(bz, 2)
                else:
                    bz = self.aux_total_bz
                ratio = 0.0
            else:
                bz = self.config.batch_size  
                ratio = self.config.revisit_policy_search_rate

            beta = self.beta_schedule.value(trained_steps)
            batch_context = ray.get(self.replay_buffer.prepare_batch_context.remote(bz, beta))
                

            # break
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                break

            new_model_index = trained_steps // self.config.target_model_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
            else:
                target_weights = None

            if self.mcts_storage.get_len() < 20:
                self.make_batch(batch_context, ratio, weights=target_weights, bz=bz)

    def make_batch(self, batch_context, ratio, weights=None, bz=None):
        
        game_name = ray.get(self.replay_buffer.get_buffer_name.remote())


        if self.multi_game:
            actions_num = self.game_info[game_name]['total_action']
            actions_id  = self.game_info[game_name]['action_id']
            # create dict for action id
            action_id_dict = {}
            for idx, item in enumerate(actions_id):
                action_id_dict[idx] = item
             

        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        for i in range(batch_size):
            game = game_lst[i]
            game_pos = game_pos_lst[i]

            _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
            # TODO:WARNING: the order of the action is very important
            if self.multi_game:
                # convert _actions based on action_id_dict
                _actions = [action_id_dict[action] for action in _actions]

            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]

            # this is action for train to recover unroll steps 
            # thus, the action space should be 18
            _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]

            obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))
            action_lst.append(_actions)
            mask_lst.append(_mask)

        re_num = int(batch_size * ratio)
        obs_lst = prepare_observation_lst(obs_lst)
        batch = [obs_lst, action_lst, mask_lst, [], [], [], indices_lst, weights_lst, make_time_lst]


        total_transitions = ray.get(self.replay_buffer.get_total_len.remote())

        if self.config.reanalyze_part == 'paper':
            if ratio == 1.0: # full reanalyze
                item = prepare_multi_target_cpu(indices_lst[:re_num], make_time_lst[:re_num], game_lst[:re_num], game_pos_lst[:re_num], total_transitions, self.config)
                item = [item, [], batch, weights, game_name]
            elif ratio == 0.0: # no reanalyze
                # TODO: this is loading the pretrained agent data as imitation learning
                #item_ov = prepare_multi_target_only_value_cpu(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:], total_transitions, self.config)
                item_ov = prepare_imitation_learning(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:], total_transitions, self.config)
                item = [[], item_ov, batch, weights, game_name]

                item = imitation_postprocess(item, self.game_info, self.config)

            else: # mix reanalyze and no reanalyze
                item = prepare_multi_target_cpu(indices_lst[:re_num], make_time_lst[:re_num], game_lst[:re_num], game_pos_lst[:re_num], total_transitions, self.config)
                item_ov = prepare_multi_target_only_value_cpu(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:], total_transitions, self.config)
                item = [item, item_ov, batch, weights, game_name]
            self.mcts_storage.push(item)
            self.count_idx += 1



def get_modified_weights(ddp_model):
    return {'.'.join(k.split('.')[1:]): v.cpu() for k, v in ddp_model.state_dict().items()}


def _train(model, target_model, latest_model, config, shared_storage, replay_buffer, batch_storage, summary_writer, game_info, replay_buffer_dict=None, aux_storage=None, multi_game=False, debug_ratio=1):
    
    # ----------------------------------------------------------------------------------
    model = model.to(config.device)
    target_model = target_model.to(config.device)

    scaler = GradScaler()

    model_params = [p for n, p in model.named_parameters()]

    # repo: https://github.com/anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum 
    opt_default_lr = [config.lr_init]
    optimizer = optim.SGD(model_params, lr=opt_default_lr[0], momentum=config.momentum, weight_decay=config.weight_decay)

    # current only support none-amp
    main_task = config.env_name
    aux_task_list = config.aux_data_list.split('/')
    # add NoFrameskip-v4 to element in aux_task_list
    aux_task_list = [aux_task + 'NoFrameskip-v4' for aux_task in aux_task_list]
    total_tasks = [main_task] + aux_task_list
    if config.amp_type == 'torch_amp':
        # get keys from replay_buffer_dict
        pcgrad = PCGradAMP(total_tasks=total_tasks, optimizer=optimizer, scaler=scaler)
    else:
        pcgrad = PCGradAMP(total_tasks=total_tasks,  optimizer=optimizer)



    model.train()
    target_model.eval()
    latest_model.eval()
    # ----------------------------------------------------------------------------------

    if config.lr_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.training_steps + config.last_steps - config.lr_warm_step
                                                               , eta_min=config.lr_min)
    else:
        scheduler = None

    if config.use_augmentation:
        config.set_transforms()

    # wait for all replay buffer to be non-empty
    print("getting replay buffer")
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_window_size//debug_ratio):
        print(ray.get(replay_buffer.get_total_len.remote()), config.start_window_size//debug_ratio)
        time.sleep(1)
        pass
    print('Begin training...')
    shared_storage.set_start_signal.remote()

    step_count = 0
    batch_count = 0
    make_time = 0.
    lr = 0.

    recent_weights = model.get_weights()
    
    num_batch = 1

    while step_count < config.training_steps + config.last_steps:
        st = time.time()
        if step_count % 1000 == 0:
            replay_buffer.remove_to_fit.remote()

        batch_list = []
        bz_list = []
        while len(batch_list) < num_batch:
            batch = batch_storage.pop()
            if batch is None:
                print('No data for muzero (target task), continue...')
                time.sleep(0.3)
                continue
            else:
                batch_list.append(batch)
                bz_list.append(batch[0].shape[0])

        # aux_offline
        num_batch_aux = len(config.aux_data_list.split('/'))
        while len(batch_list) < (num_batch+num_batch_aux):
            aux_batch = aux_storage.pop()
            if aux_batch is None:
                print('No data for muzero (aux task), continue...')
                time.sleep(0.3)
                continue
            else:
                batch_list.append(aux_batch)
                bz_list.append(aux_batch[0].shape[0])

        shared_storage.incr_counter.remote()

        if config.lr_type == 'cosine':
            lr = adjust_lr(config, optimizer, step_count, scheduler, opt_default_lr, config.lr_min)
        else:
            lr = adjust_lr(config, optimizer, step_count, scheduler, opt_default_lr)

        # SET MUZERO @100
        if step_count % config.checkpoint_interval == 0: 
            shared_storage.set_weights.remote(model.get_weights())
            if config.use_latest_model: 
                soft_update(latest_model, model.detach(), tau=1)
                shared_storage.set_latest_weights.remote(latest_model.get_weights())

        # SET TARGET MODEL @200
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights)
            recent_weights = model.get_weights()

        # UPDATE MUZERO
        if config.amp_type == 'torch_amp':
            if step_count >= 1:
                scaler = scaler_prev
            log_data = update_weights(model, batch_list, optimizer, replay_buffer, config, scaler, game_info=game_info, \
                                        replay_buffer_dict=replay_buffer_dict, multi_game=multi_game, bz_list=bz_list, step_count=step_count, pcgrad=pcgrad)
            scaler_prev = log_data[1]
        else:
            log_data = update_weights(model, batch_list, optimizer, replay_buffer, config, scaler, game_info=game_info, \
                                        replay_buffer_dict=replay_buffer_dict, multi_game=multi_game, bz_list=bz_list, step_count=step_count, pcgrad=pcgrad)

        # LOG @ 1000
        if step_count % config.log_interval == 0: 
            _log(config, step_count, log_data[0], replay_buffer, lr, shared_storage, game_name=log_data[2])

        step_count += 1

        # SAVE MUZERO @10000
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

        # PRINT @ 20
        if step_count % 20 == 0:
            print('Tloop={:.2f}, BatchQ={}, LR={}, BZ={}'.format(time.time()-st, batch_storage.get_len(), lr, sum(bz_list)))

    shared_storage.set_weights.remote(model.get_weights())
    return model.get_weights()


@ray.remote(num_gpus=0.5)
def _test(config, shared_storage, game_info, eval_game, multi_game=False):
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
    episodes = 0
    while True:
        counter = ray.get(shared_storage.get_counter.remote())
        if counter >= config.training_steps + config.last_steps:
            time.sleep(30)
            break
        if counter >= config.test_interval * episodes:
            episodes += 1
            test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            test_model.eval()

            test_score, _ = test(config, test_model, counter, config.test_episodes, 'cuda', False, save_video=False, \
                                game_info=game_info, eval_game=eval_game, multi_game=multi_game)
            # mean_score = sum(test_score) / len(test_score)
            mean_score = test_score.mean()
            std_score = np.sqrt(test_score.var())
            if mean_score >= best_test_score:
                best_test_score = mean_score
                torch.save(test_model.state_dict(), config.model_path)
            
            mean_name = 'mean_score'
            std_name = 'std_score'
            test_log = {
                mean_name: mean_score,
                std_name: std_score,
            }

            shared_storage.add_test_log.remote(counter, test_log)
            print('Step {}, test scores: {}'.format(counter, test_score))

        time.sleep(30)


def train(config, summary_writer, model_path=None, game_info=None, multi_game=False, aux_data_dict=None, data_save_path=None):

    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    latest_model = config.get_uniform_network()


    if model_path:
        print('resume model from path: {}'.format(model_path))

        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)
        latest_model.load_state_dict(weights)
    else:
        assert False, "required multitask pretrained model for online finetuning"

    storage = SharedStorage.remote(model, target_model, latest_model)
    
    batch_storage = BatchStorage(15, 20)
    mcts_storage = MCTS_Storage(20, 25)


    # Optional: load pretrained buffer
    replay_buffer_dict = {}
    aux_rb_list = []
    for i, (k, v) in enumerate(aux_data_dict.items()):
        load_count = 0

        # copy config 
        config_copy = copy.deepcopy(config)
        config_copy.set_game(k, set_action=False)

        rb = ReplayBuffer.remote(replay_buffer_id=i, config=config_copy, buffer_name=k)
        # reverse list
        for data_path in v:
            for idx in range(0, 8):
                rb.load_files.remote(data_path, idx)
                load_count += 1
        aux_rb_list.append(rb)
        replay_buffer_dict[k] = rb
        
        # wait until self._eps_collected == i
        while not (ray.get(rb.episodes_collected.remote()) == load_count):
            print(ray.get(rb.episodes_collected.remote()),load_count)
            time.sleep(1)
            pass
    len_aux_rb_list = len(aux_rb_list)
    aux_storage = MCTS_Storage(20, 25)

    # Online ReplyBuffer
    b_name = config.env_name
    replay_buffer = ReplayBuffer.remote(replay_buffer_id=0, config=config, buffer_name=b_name, data_path=data_save_path)
    replay_buffer_dict[b_name] = replay_buffer

    workers = []

    # self-play
    data_workers = [DataWorker.remote(rank, config, storage, replay_buffer,game_info=game_info, multi_game=multi_game) for rank in range(0, config.num_actors)]
    workers += [worker.run_multi.remote() for worker in data_workers]

    # add offline cpu worker
    if config.aux_offline:
        # Offline CPU Worker => aux_storage
        aux_workers = [BatchWorker_CPU.remote(idx, aux_rb_list[idx%len_aux_rb_list], storage, aux_storage, config, game_info, multi_game=multi_game, start_now=True) for idx in range(len_aux_rb_list*3)]
        workers += [aux_worker.run.remote() for aux_worker in aux_workers]

    # reanalyze
    cpu_workers = [BatchWorker_CPU.remote(idx, replay_buffer, storage, mcts_storage, config, game_info, multi_game=multi_game) for idx in range(config.cpu_actor)]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    gpu_workers = [PrepareTargetGpuActor.remote(idx, replay_buffer, config, mcts_storage, storage, batch_storage, game_info=game_info, multi_game=multi_game) for idx in range(config.gpu_actor)]
    workers += [gpu_worker.prepare_target_gpu.remote() for gpu_worker in gpu_workers]

    # evaluation workers
    workers += [_test.remote(config, storage, game_info=game_info, eval_game=config.env_test, multi_game=multi_game)]

    # train
    final_weights = _train(model, target_model, latest_model, config, storage, replay_buffer, batch_storage, summary_writer, \
                            game_info=game_info, replay_buffer_dict=replay_buffer_dict, aux_storage=aux_storage, multi_game=multi_game)

    ray.wait(workers)
    print('Training over...')
    return model, final_weights

