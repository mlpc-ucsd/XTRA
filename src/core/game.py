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


class Player(object):
    def __init__(self, id=1):
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id


class Game:
    def __init__(self, env, action_space_size: int, discount: float, config=None):
        self.env = env
        self.action_space_size = action_space_size
        self.discount = discount
        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class GameHistory:
    """
        Store only usefull information of a self-play game.
        """

    def __init__(self, action_space, max_length=200, config=None):
        self.action_space = action_space
        self.max_length = max_length
        self.config = config

        self.stacked_observations = config.stacked_observations
        self.discount = config.discount
        self.action_space_size = config.action_space_size

        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []

    def init(self, init_observations):
        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []
        self.target_values = []
        self.target_rewards = []
        self.target_policies = []

        assert len(init_observations) == self.stacked_observations

        for observation in init_observations:
            self.obs_history.append(copy.deepcopy(observation))

    def pad_over(self, next_block_observations, next_block_rewards, next_block_root_values, next_block_child_visits):
        assert len(next_block_observations) <= self.config.num_unroll_steps
        assert len(next_block_child_visits) <= self.config.num_unroll_steps
        assert len(next_block_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_block_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # notice: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_block_observations:
            self.obs_history.append(copy.deepcopy(observation))

        for reward in next_block_rewards:
            self.rewards.append(reward)

        for value in next_block_root_values:
            self.root_values.append(value)

        for child_visits in next_block_child_visits:
            self.child_visits.append(child_visits)

    def is_full(self):
        return self.__len__() >= self.max_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def load_file(self, path, load_target=False):
        assert os.path.exists(path)

        self.actions = np.load(os.path.join(path, 'actions.npy'))#.tolist()
        obs_history = np.load(os.path.join(path, 'obs.npy'), allow_pickle=True)
        self.obs_history = ray.put(obs_history)
        self.rewards = np.load(os.path.join(path, 'reward.npy'))
        self.child_visits = np.load(os.path.join(path, 'visits.npy'))
        self.root_values = np.load(os.path.join(path, 'root.npy'))
        # last_observations = [self.obs_history[-1] for i in range(self.config.num_unroll_steps)]
        # self.obs_history = np.concatenate((self.obs_history, last_observations))

        if load_target:
            self.target_values = np.load(os.path.join(path, 'target_values.npy'))
            self.target_rewards = np.load(os.path.join(path, 'target_rewards.npy'))
            self.target_policies = np.load(os.path.join(path, 'target_policies.npy'))

    def save_file(self, path, save_target=False):
        if not os.path.exists(path):
            os.mkdir(path)

        np.save(os.path.join(path, 'actions.npy'), np.array(self.actions))
        np.save(os.path.join(path, 'obs.npy'), np.array(ray.get(self.obs_history)), allow_pickle=True)
        np.save(os.path.join(path, 'reward.npy'), np.array(self.rewards))
        np.save(os.path.join(path, 'visits.npy'), np.array(self.child_visits))
        np.save(os.path.join(path, 'root.npy'), np.array(self.root_values))
        if save_target:
            np.save(os.path.join(path, 'target_values.npy'), np.array(self.target_values))
            np.save(os.path.join(path, 'target_rewards.npy'), np.array(self.target_rewards))
            np.save(os.path.join(path, 'target_policies.npy'), np.array(self.target_policies))

    def append(self, action, obs, reward):
        self.actions.append(action)
        self.obs_history.append(obs)
        self.rewards.append(reward)

    def obs_object(self):
        return self.obs_history

    def obs(self, i, extra_len=0, padding=False):
        frames = ray.get(self.obs_history)[i:i + self.stacked_observations + extra_len]
        if padding:
            pad_len = self.stacked_observations + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def zero_obs(self):
        return [np.ones((96, 96, self.config.image_channel), dtype=np.uint8) for _ in range(self.stacked_observations)]

    def step_obs(self, index=None):
        if index is None:
            index = len(self.rewards)
        try:
            frames = self.obs_history[index:index + self.stacked_observations]
        except:
            frames = ray.get(self.obs_history)[index:index + self.stacked_observations]
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def get_targets(self, i):
        return self.target_values[i], self.target_rewards[i], self.target_policies[i]

    def game_over(self):
        self.rewards = np.array(self.rewards)
        self.obs_history = ray.put(np.array(self.obs_history))
        self.actions = np.array(self.actions)
        self.child_visits = np.array(self.child_visits)
        self.root_values = np.array(self.root_values)

    def store_search_stats(self, visit_counts, root_value, idx: int = None, set_flag=False):
        if set_flag:
            self.child_visits.setflags(write=1)
            self.root_values.setflags(write=1)

        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visits.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_values.append(root_value)
        else:
            self.child_visits[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_values[idx] = root_value

        if set_flag:
            self.child_visits.setflags(write=0)
            self.root_values.setflags(write=0)

    def action_history(self, idx=None) -> list:
        if idx is None:
            return self.actions
        else:
            return self.actions[:idx]

    # def to_play(self) -> Player:
    #     return Player()

    def __len__(self):
        return len(self.actions)


def prepare_multi_target_only_value(indices, games, state_index_lst, total_transitions, config, model):
    batch_values, batch_reward_sums, batch_policies = [], [], []

    zero_obs = games[0].zero_obs()
    device = next(model.parameters()).device
    obs_lst = []
    value_mask = []

    with torch.no_grad():
        td_steps_lst = []
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

                obs_lst.append(obs)

        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst)
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = torch.from_numpy(obs_lst[beg_index:end_index]).to(device).float() / 255.0
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)
            network_output.append(m_output)

        value_pool, reward_sum_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
        if config.off_correction:
            roots = cytree.Roots(len(obs_lst), config.action_space_size, config.num_simulations)
            # noises = [
            #     np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            #         np.float32).tolist()
            #     for _ in range(len(obs_lst))]
            # roots.prepare(config.root_exploration_fraction, noises, reward_sum_pool, policy_logits_pool)
            roots.prepare_no_noise(reward_sum_pool.tolist(), policy_logits_pool.tolist())
            MCTS(config).run_multi(roots, model, hidden_state_roots, reward_hidden_roots)

            roots_values = roots.get_values()
            value_lst = np.array(roots_values)
        else:
            value_lst = value_pool

        # value_lst = concat_output_value(network_output)
        value_lst = value_lst.reshape(-1) * (np.array([config.discount for _ in range(batch_num)]) ** td_steps_lst)
        # get last value
        value_lst = value_lst * np.array(value_mask)
        value_lst = value_lst.tolist()

        value_index = 0
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            target_values = []
            target_reward_sums = []

            reward_sum = 0.0
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + td_steps_lst[value_index]
                for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    reward_sum += game.rewards[current_index]
                    target_reward_sums.append(reward_sum)
                else:
                    target_values.append(0)
                    target_reward_sums.append(reward_sum)
                value_index += 1

            batch_values.append(target_values)
            batch_reward_sums.append(target_reward_sums)

    # for policy
    policy_mask = []  # 0 -> out of traj, 1 -> old policy
    for game, state_index in zip(games, state_index_lst):
        traj_len = len(game)
        target_policies = []

        for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
            if current_index < traj_len:
                target_policies.append(game.child_visits[current_index])
                policy_mask.append(1)
            else:
                target_policies.append([0 for _ in range(config.action_space_size)])
                policy_mask.append(0)

        batch_policies.append(target_policies)

    return batch_values, batch_reward_sums, batch_policies


def prepare_multi_target(replay_buffer, indices, make_time, games, state_index_lst, total_transitions, config, model):
    batch_values, batch_reward_sums, batch_policies = [], [], []
    zero_obs = games[0].zero_obs()

    device = next(model.parameters()).device
    obs_lst = []
    value_mask = []

    with torch.no_grad():
        td_steps_lst = []
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

                obs_lst.append(obs)

        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst)
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = torch.from_numpy(obs_lst[beg_index:end_index]).to(device).float() / 255.0
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)
            network_output.append(m_output)

        value_pool, reward_sum_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(
            network_output)
        if config.off_correction:
            roots = cytree.Roots(len(obs_lst), config.action_space_size, config.num_simulations)
            # noises = [
            #     np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            #         np.float32).tolist()
            #     for _ in range(len(obs_lst))]
            # roots.prepare(config.root_exploration_fraction, noises, reward_sum_pool, policy_logits_pool)
            roots.prepare_no_noise(reward_sum_pool.tolist(), policy_logits_pool.tolist())
            MCTS(config).run_multi(roots, model, hidden_state_roots, reward_hidden_roots)

            roots_values = roots.get_values()
            value_lst = np.array(roots_values)
        else:
            value_lst = value_pool

        # value_lst = concat_output_value(network_output)
        value_lst = value_lst.reshape(-1) * (np.array([config.discount for _ in range(batch_num)]) ** td_steps_lst)
        # get last value
        value_lst = value_lst * np.array(value_mask)
        value_lst = value_lst.tolist()

        value_index = 0
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            target_values = []
            target_reward_sums = []

            reward_sum = 0.0
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + td_steps_lst[value_index]
                for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    reward_sum += game.rewards[current_index]
                    target_reward_sums.append(reward_sum)
                else:
                    target_values.append(0)
                    target_reward_sums.append(reward_sum)
                value_index += 1

            batch_values.append(target_values)
            batch_reward_sums.append(target_reward_sums)

        # for policy
        obs_lst = []
        policy_mask = []  # 0 -> out of traj, 1 -> new policy
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)

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
                obs_lst.append(obs)

        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst)
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = torch.from_numpy(obs_lst[beg_index:end_index]).to(device).float() / 255.0
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)
            network_output.append(m_output)

        _, reward_sum_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
        reward_sum_pool = reward_sum_pool.squeeze().tolist()
        policy_logits_pool = policy_logits_pool.tolist()

        roots = cytree.Roots(len(obs_lst), config.action_space_size, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(np.float32).tolist() for _ in range(len(obs_lst))]
        roots.prepare(config.root_exploration_fraction, noises, reward_sum_pool, policy_logits_pool)

        MCTS(config).run_multi(roots, model, hidden_state_roots, reward_hidden_roots)

        roots_distributions = roots.get_distributions()
        roots_values = roots.get_values()
        policy_index = 0
        game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times = [], [], [], [], []
        for game, state_index, game_idx, mt in zip(games, state_index_lst, indices, make_time):
            target_policies = []

            current_index_lst, distributions_lst, value_lst, make_time_lst = [], [], [], []
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                distributions, value = roots_distributions[policy_index], roots_values[policy_index]

                if policy_mask[policy_index] == 0:
                    target_policies.append([0 for _ in range(config.action_space_size)])
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
            if config.write_back:
                game_idx_lst.append(game_idx)
                current_index_lsts.append(current_index_lst)
                distributions_lsts.append(distributions_lst)
                value_lsts.append(value_lst)
                make_times.append(make_time_lst)

        if config.write_back:
            replay_buffer.update_games.remote(game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times)

    return batch_values, batch_reward_sums, batch_policies
