from typing import List
from .utils import retype_observation, prepare_observation, prepare_observation_lst, str_to_arr
from core.mcts import MCTS
import core.ctree.cytree as cytree
from core.model import concat_output, concat_output_value
from torch.cuda.amp import autocast as autocast
import numpy as np
import torch
import os
import ray
import copy
import time
from ray.util.queue import Queue
try:
    from apex import amp
except:
    pass


class MCTS_Storage(object):
    def __init__(self, threshold=30, size=40):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, item):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(item)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()


# this is re-analysis part
def prepare_multi_target_cpu(indices, make_time, games, state_index_lst, total_transitions, config):
    zero_obs = games[0].zero_obs()
    obs_lst1 = []
    value_mask = []

    td_steps_lst = []
    with torch.no_grad():
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)

            delta_td = (total_transitions - idx) // config.auto_td_steps
            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, 5).astype(np.int)

            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # obs = game.obs(bootstrap_index)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                    # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
                else:
                    value_mask.append(0)
                    obs = zero_obs

                obs_lst1.append(obs)

        # for policy
        obs_lst2 = []
        policy_mask = []  # 0 -> out of traj, 1 -> new policy
        rewards, child_visits, traj_lens = [], [], []
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            traj_lens.append(traj_len)
            rewards.append(game.rewards)
            child_visits.append(game.child_visits)
            game_obs = game.obs(state_index, config.num_unroll_steps)
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                if current_index < traj_len:
                    policy_mask.append(1)
                    # obs = game.obs(current_index)
                    beg_index = current_index - state_index
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                    # assert (np.array(obs) == np.array(game.obs(current_index))).all()
                else:
                    policy_mask.append(0)
                    obs = zero_obs
                obs_lst2.append(obs)

    obs_lst1, obs_lst2 = ray.put(obs_lst1), ray.put(obs_lst2)
    item = [obs_lst1, obs_lst2, value_mask, policy_mask, state_index_lst, indices, make_time, rewards, child_visits, traj_lens, td_steps_lst]
    return item

# this is not re-analysis part
def prepare_multi_target_only_value_cpu(indices, games, state_index_lst, total_transitions, config):
    zero_obs = games[0].zero_obs()
    obs_lst = []
    value_mask = []
    rewards_ov = []
    child_visits_ov = []
    traj_lens_ov = []

    td_steps_lst = []
    for game, state_index, idx in zip(games, state_index_lst, indices):
        traj_len = len(game)
        traj_lens_ov.append(traj_len)
        delta_td = (total_transitions - idx) // config.auto_td_steps
        td_steps = config.td_steps - delta_td
        td_steps = np.clip(td_steps, 1, 5).astype(np.int)

        game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
        rewards_ov.append(game.rewards)
        child_visits_ov.append(game.child_visits)
        for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
            td_steps_lst.append(td_steps)
            bootstrap_index = current_index + td_steps

            if bootstrap_index < traj_len:
                value_mask.append(1)
                # obs = game.obs(bootstrap_index)
                beg_index = bootstrap_index - (state_index + td_steps)
                end_index = beg_index + config.stacked_observations
                obs = game_obs[beg_index:end_index]
                # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
            else:
                value_mask.append(0)
                obs = zero_obs

            obs_lst.append(obs)

    obs_lst = ray.put(obs_lst)
    item = [obs_lst, value_mask, state_index_lst, rewards_ov, child_visits_ov, traj_lens_ov, td_steps_lst]
    return item

# this is imitatio re-analysis part
def prepare_imitation_learning(indices, games, state_index_lst, total_transitions, config):
    zero_obs = games[0].zero_obs()
    obs_lst = []
    value_mask = []
    rewards_ov = []
    child_visits_ov = []
    root_values_ov = []
    traj_lens_ov = []

    td_steps_lst = []
    for game, state_index, idx in zip(games, state_index_lst, indices):
        traj_len = len(game)
        traj_lens_ov.append(traj_len)
        #delta_td = (total_transitions - idx) // config.auto_td_steps
        #td_steps = config.td_steps - delta_td
        #td_steps = np.clip(td_steps, 1, 5).astype(np.int)
        td_steps = 5
        game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
        rewards_ov.append(game.rewards)
        child_visits_ov.append(game.child_visits)
        root_values_ov.append(game.root_values)
        for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
            td_steps_lst.append(td_steps)
            bootstrap_index = current_index + td_steps

            if bootstrap_index < traj_len:
                value_mask.append(1)
                # obs = game.obs(bootstrap_index)
                beg_index = bootstrap_index - (state_index + td_steps)
                end_index = beg_index + config.stacked_observations
                obs = game_obs[beg_index:end_index]
                # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
            else:
                value_mask.append(0)
                obs = zero_obs

            obs_lst.append(obs)

    obs_lst = ray.put(obs_lst)
    item = [obs_lst, value_mask, state_index_lst, rewards_ov, child_visits_ov, traj_lens_ov, td_steps_lst, root_values_ov]
    return item

def imitation_postprocess(input, game_info, config):
    
    item, item_ov, batch, _, game_name = input

    actions_num = game_info[game_name]['total_action']
    actions_id  = game_info[game_name]['action_id']

    batch.append(game_name)

    # TODO NON_RE: _prepare_reward_value 
    assert len(item) == 0
    if len(item_ov) > 0:
        obs_lst_ov, value_mask_ov, state_index_lst_ov, rewards_ov, child_visits_ov, traj_lens_ov, td_steps_lst_ov, root_values_ov = item_ov
        obs_lst_ov = ray.get(obs_lst_ov)

        batch_values_ov, batch_reward_sums_ov, batch_policies_ov = [], [], []

        # for game, state_index in zip(games, state_index_lst):
        for traj_len_ov, reward_ov, root_value_ov, state_index in zip(traj_lens_ov, rewards_ov, root_values_ov, state_index_lst_ov):
            # traj_len = len(game)
            target_values = []
            target_reward_sums = []

            reward_sum = 0.0
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + 5
                # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                reward4value = 0.0
                for i, reward in enumerate(reward_ov[current_index:bootstrap_index]):
                    reward4value += reward * config.discount ** i

                if current_index < traj_len_ov:
                    target_values.append(root_value_ov[current_index]+reward4value)
                    reward_sum += reward_ov[current_index]
                    target_reward_sums.append(reward_sum)
                else:
                    target_values.append(0)
                    target_reward_sums.append(reward_sum)

            batch_values_ov.append(target_values)
            batch_reward_sums_ov.append(target_reward_sums)

        # TODO NON_RE:  _prepare_policy_non_re 
        # for policy
        policy_mask = []  # 0 -> out of traj, 1 -> old policy
        # for game, state_index in zip(games, state_index_lst):
        for traj_len_ov, child_visit_ov, state_index in zip(traj_lens_ov, child_visits_ov, state_index_lst_ov):
            # traj_len = len(game)
            target_policies = []

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                if current_index < traj_len_ov:
                    target_policies.append(child_visit_ov[current_index])
                    policy_mask.append(1)
                else:
                    target_policies.append([0 for _ in range(actions_num)])
                    policy_mask.append(0)

            batch_policies_ov.append(target_policies)


        if len(item) > 0:
            batch[3].append(batch_reward_sums)
            batch[4].append(batch_values)
            batch[5].append(batch_policies)
        if len(item_ov) > 0:
            batch[3].append(batch_reward_sums_ov)
            batch[4].append(batch_values_ov)
            batch[5].append(batch_policies_ov)

        for i in range(len(batch)):
            if i in range(3, 6):
                batch[i] = np.concatenate(batch[i])
            else:
                batch[i] = np.asarray(batch[i])

        return batch

@ray.remote(num_gpus=0.125)
class PrepareTargetGpuActor(object):
    def __init__(self, gpu_worker_id, replay_buffer, config, mcts_storage, storage, batch_storage, game_info=None, multi_game=False):
        self.replay_buffer = replay_buffer
        self.config = config
        self.worker_id = gpu_worker_id

        self.model = config.get_uniform_network()
        self.model.to(config.device)
        self.model.eval()

        self.mcts_storage = mcts_storage
        self.storage = storage
        self.batch_storage = batch_storage

        self.last_model_index = 0
        self.print_cnt = 0

        self.game_info = game_info
        self.count_idx = 0
        self.multi_game = multi_game
    def prepare_target_gpu(self):

        start = False
        while True:
            # waiting for start signal
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(0.1)
                continue

            self._prepare_target_gpu()

    def _prepare_target_gpu(self):

        input = self.mcts_storage.pop()
        if input is None:
            self.print_cnt += 1
            if self.print_cnt % 3 == 0:
                print('waiting for ...[CPU]')
                pass
            time.sleep(1)
        else:
            item, item_ov, batch, target_weights, game_name = input     
            if target_weights is not None:
                self.model.load_state_dict(target_weights)
                self.model.to(self.config.device)
                self.model.eval()

            actions_num = self.game_info[game_name]['total_action']
            actions_id  = self.game_info[game_name]['action_id']

            batch.append(game_name)

            # TODO NON_RE: _prepare_reward_value 
            if len(item_ov) > 0:
                obs_lst_ov, value_mask_ov, state_index_lst_ov, rewards_ov, child_visits_ov, traj_lens_ov, td_steps_lst_ov = item_ov
                obs_lst_ov = ray.get(obs_lst_ov)

                batch_values_ov, batch_reward_sums_ov, batch_policies_ov = [], [], []
                with torch.no_grad():
                    device = next(self.model.parameters()).device
                    batch_num = len(obs_lst_ov)
                    obs_lst_ov = prepare_observation_lst(obs_lst_ov)
                    m_batch = self.config.target_infer_size
                    slices = batch_num // m_batch
                    if m_batch * slices < batch_num:
                        slices += 1
                    network_output = []
                    for i in range(slices):
                        beg_index = m_batch * i
                        end_index = m_batch * (i + 1)
                        m_obs = torch.from_numpy(obs_lst_ov[beg_index:end_index]).to(device).float() / 255.0
                        if self.config.amp_type == 'torch_amp':
                            with autocast():
                                m_output = self.model.initial_inference(m_obs, actions_id)
                        else:
                            m_output = self.model.initial_inference(m_obs, actions_id)
                        network_output.append(m_output)

                    value_lst = concat_output_value(network_output)
                    value_lst = value_lst.reshape(-1) * (np.array([self.config.discount for _ in range(batch_num)]) ** td_steps_lst_ov)
                    # get last value
                    value_lst = value_lst * np.array(value_mask_ov)
                    value_lst = value_lst.tolist()

                    value_index = 0
                    # for game, state_index in zip(games, state_index_lst):
                    for traj_len_ov, reward_ov, state_index in zip(traj_lens_ov, rewards_ov, state_index_lst_ov):
                        # traj_len = len(game)
                        target_values = []
                        target_reward_sums = []

                        reward_sum = 0.0
                        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                            bootstrap_index = current_index + td_steps_lst_ov[value_index]
                            # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                            for i, reward in enumerate(reward_ov[current_index:bootstrap_index]):
                                value_lst[value_index] += reward * self.config.discount ** i

                            if current_index < traj_len_ov:
                                target_values.append(value_lst[value_index])
                                reward_sum += reward_ov[current_index]
                                target_reward_sums.append(reward_sum)
                            else:
                                target_values.append(0)
                                target_reward_sums.append(reward_sum)
                            value_index += 1

                        batch_values_ov.append(target_values)
                        batch_reward_sums_ov.append(target_reward_sums)

                    # TODO NON_RE:  _prepare_policy_non_re 
                    # for policy
                    policy_mask = []  # 0 -> out of traj, 1 -> old policy
                    # for game, state_index in zip(games, state_index_lst):
                    for traj_len_ov, child_visit_ov, state_index in zip(traj_lens_ov, child_visits_ov, state_index_lst_ov):
                        # traj_len = len(game)
                        target_policies = []

                        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                            if current_index < traj_len_ov:
                                target_policies.append(child_visit_ov[current_index])
                                policy_mask.append(1)
                            else:
                                target_policies.append([0 for _ in range(actions_num)])
                                policy_mask.append(0)

                        batch_policies_ov.append(target_policies)

            # TODO RE: _prepare_reward_value
            if len(item) > 0:
                obs_lst1, obs_lst2, value_mask, policy_mask, state_index_lst, indices, make_time, rewards, child_visits, traj_lens, td_steps_lst = item
                obs_lst1 = ray.get(obs_lst1)
                obs_lst2 = ray.get(obs_lst2)
                batch_values, batch_reward_sums, batch_policies = [], [], [] 

                with torch.no_grad():
                    device = next(self.model.parameters()).device
                    batch_num = len(obs_lst1)
                    obs_lst = prepare_observation_lst(obs_lst1)
                    m_batch = self.config.target_infer_size
                    slices = batch_num // m_batch
                    if m_batch * slices < batch_num:
                        slices += 1
                    network_output = []
                    for i in range(slices):
                        beg_index = m_batch * i
                        end_index = m_batch * (i + 1)
                        m_obs = torch.from_numpy(obs_lst[beg_index:end_index]).to(device).float() / 255.0
                        if self.config.amp_type == 'torch_amp':
                            with autocast():
                                m_output = self.model.initial_inference(m_obs, actions_id)
                        else:
                            m_output = self.model.initial_inference(m_obs, actions_id)
                        network_output.append(m_output)

                    value_lst = concat_output_value(network_output)
                    value_lst = value_lst.reshape(-1) * (np.array([self.config.discount for _ in range(batch_num)]) ** td_steps_lst)
                    # get last value
                    value_lst = value_lst * np.array(value_mask)
                    value_lst = value_lst.tolist()

                    value_index = 0
                    # for game, state_index in zip(games, state_index_lst):
                    for traj_len, reward_t, state_index in zip(traj_lens, rewards, state_index_lst):
                        # traj_len = len(game)
                        target_values = []
                        target_reward_sums = []

                        reward_sum = 0.0
                        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                            bootstrap_index = current_index + td_steps_lst[value_index]
                            for i, reward in enumerate(reward_t[current_index:bootstrap_index]):
                                value_lst[value_index] += reward * self.config.discount ** i

                            if current_index < traj_len:
                                target_values.append(value_lst[value_index])
                                reward_sum += reward_t[current_index]
                                target_reward_sums.append(reward_sum)
                            else:
                                target_values.append(0)
                                target_reward_sums.append(reward_sum)
                            value_index += 1

                        batch_values.append(target_values)
                        batch_reward_sums.append(target_reward_sums)

                    # TODO RE: _prepare_policy
                    device = next(self.model.parameters()).device
                    batch_num = len(obs_lst2)
                    obs_lst = prepare_observation_lst(obs_lst2)
                    m_batch = self.config.target_infer_size
                    slices = batch_num // m_batch
                    if m_batch * slices < batch_num:
                        slices += 1
                    network_output = []

                    for i in range(slices):
                        beg_index = m_batch * i
                        end_index = m_batch * (i + 1)
                        m_obs = torch.from_numpy(obs_lst[beg_index:end_index]).to(device).float() / 255.0
                        if self.config.amp_type == 'torch_amp':
                            with autocast():
                                m_output = self.model.initial_inference(m_obs, actions_id, keep_tensor=False)
                        else:
                            m_output = self.model.initial_inference(m_obs, actions_id)
                        network_output.append(m_output)

                    # MCTS rollout
                    _, reward_sum_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
                    reward_sum_pool = reward_sum_pool.squeeze().tolist()
                    policy_logits_pool = policy_logits_pool.tolist()
                    roots = cytree.Roots(len(obs_lst), actions_num, self.config.num_simulations)
                    noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * actions_num).astype(np.float32).tolist() for _ in range(len(obs_lst))]
                    roots.prepare(self.config.root_exploration_fraction, noises, reward_sum_pool, policy_logits_pool)
                    MCTS(self.config).run_multi(roots, self.model, hidden_state_roots, reward_hidden_roots, actions_id, self.multi_game)
                    roots_distributions = roots.get_distributions()
                    roots_values = roots.get_values()


                    # Assignment to [bz, step, : ]
                    policy_index = 0 
                    game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times = [], [], [], [], []
                    for state_index, game_idx, mt in zip(state_index_lst, indices, make_time):
                        target_policies = []

                        current_index_lst, distributions_lst, value_lst, make_time_lst = [], [], [], []
                        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1): # self.config.num_unroll_steps + 1  = 6
                            distributions, value = roots_distributions[policy_index], roots_values[policy_index]

                            if policy_mask[policy_index] == 0:
                                target_policies.append([0 for _ in range(actions_num)])
                            else:
                                # game.store_search_stats(distributions, value, current_index)
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                target_policies.append(policy)
                                    
                                current_index_lst.append(current_index)
                                distributions_lst.append(distributions)
                                value_lst.append(value)
                                make_time_lst.append(mt)

                            policy_index += 1

                        batch_policies.append(target_policies)

                        if self.config.write_back:
                            game_idx_lst.append(game_idx)
                            current_index_lsts.append(current_index_lst)
                            distributions_lsts.append(distributions_lst)
                            value_lsts.append(value_lst)
                            make_times.append(make_time_lst)

                    if self.config.write_back:
                        self.replay_buffer.update_games.remote(game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times)

            if len(item) > 0:
                batch[3].append(batch_reward_sums)
                batch[4].append(batch_values)
                batch[5].append(batch_policies)
            if len(item_ov) > 0:
                batch[3].append(batch_reward_sums_ov)
                batch[4].append(batch_values_ov)
                batch[5].append(batch_policies_ov)

            for i in range(len(batch)):
                if i in range(3, 6):
                    batch[i] = np.concatenate(batch[i])
                else:
                    batch[i] = np.asarray(batch[i])

            self.batch_storage.push(batch)
            self.count_idx += 1
