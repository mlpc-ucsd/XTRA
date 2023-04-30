import numpy as np
import ray
import torch
import random
import time
import copy
import math
import os
from tqdm import tqdm
from ray.util.multiprocessing import Pool

from .game import GameHistory, prepare_multi_target
from .utils import profile, prepare_observation_lst


@ray.remote
class ReplayBuffer(object):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """

    def __init__(self, replay_buffer_id, make_dataset=False, reset_prior=True, config=None, data_path=None, buffer_name=None):
        self.config = config
        self.soft_capacity = config.window_size
        self.batch_size = config.batch_size
        self.make_dataset = make_dataset
        self.replay_buffer_id = replay_buffer_id
        self.reset_prior = reset_prior
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_look_up = []
        self.tail_index = []
        self.tail_len = 5
        self.tail_ratio = self.config.tail_ratio

        self._eps_collected = 0
        self.buffer_len = 0
        self.base_idx = 0
        self._alpha = config.priority_prob_alpha
        # self._beta = config.priority_prob_beta
        self.transition_top = int(config.transition_num * 10 ** 6)
        self.clear_time = 0

        if buffer_name is None:
            print('Create new replay buffer')
            #assert False
        else:
            self.buffer_name = buffer_name
        self.datapath = data_path

    def random_init_trajectory(self, num):
        assert False 

        env = self.config.new_game(self.config.seed)
        obs_ori = env.reset()

        def random_distribution(num, dim):
            res = np.zeros(dim)
            for i in range(num):
                index = np.random.randint(0, dim)
                res[index] += 1
            return res

        print('Fill random data into replay')
        for _ in tqdm(range(num)):
            game_history = GameHistory(env.env.action_space, max_length=self.config.history_length, config=self.config)
            game_history.init([obs_ori for _ in range(self.config.stacked_observations)])

            traj_len = self.config.history_length
            distributions = np.ones((traj_len, self.config.action_space_size))
            values = np.random.random(traj_len)
            actions = np.zeros(traj_len, dtype=np.int)
            priority = np.random.random(traj_len)
            reward = 0
            end_tag = True
            for i in range(traj_len):
                # distributions = random_distribution(self.config.num_simulations, self.config.action_space_size)

                game_history.store_search_stats(distributions[i], values[i])
                game_history.append(actions[i], obs_ori, reward)
            game_history.game_over()
            self.save_game(game_history, end_tag, priority)

    def load_files(self, path=None, id=None):
        if path is None:
            path = self.config.exp_path
        if id is None:
            dir = os.path.join(path, 'replay', str(self.replay_buffer_id))
        else: 
            dir = os.path.join(path, 'replay', str(id))
        print('Loading from ', dir, ' ...')
        #try:
        assert os.path.exists(dir)
        #except:
        #    print(dir)
        priorities = np.load(os.path.join(dir, 'prior.npy'))
        if self.reset_prior:
            priorities = np.ones_like(priorities)
        if len(self.priorities) == 0:
            self.priorities = priorities
        else:
            self.priorities = np.concatenate((self.priorities, priorities))

        game_look_up = np.load(os.path.join(dir, 'game_look_up.npy')).tolist()
        if len(self.game_look_up) == 0:
            self.game_look_up = game_look_up
        else:
            game_look_up = [[g[0]+self.buffer_len, g[1] ]for g in game_look_up]
            self.game_look_up += game_look_up

        _, buffer_len = np.load(os.path.join(dir, 'utils.npy')).tolist()
        if self.buffer_len == 0:
            self.buffer_len = buffer_len
        else:
            self.buffer_len += buffer_len
        self._eps_collected += 1
        assert len(self.priorities) == len(self.game_look_up)

        try:
            assert self.game_look_up[-1][0] == self.base_idx + self.buffer_len - 1
        except:
            print('Errror: ', self.game_look_up[-1][0], self.base_idx + self.buffer_len - 1)
            

        env = self.config.new_game(0)
        for i in tqdm(range(buffer_len)):
            game = GameHistory(env.env.action_space, max_length=self.config.history_length, config=self.config)
            path = os.path.join(dir, str(i))
            game.load_file(path)

            self.buffer.append(game)
        print('Load Over.')
        last_traj_act_len = len(self.buffer[-1].actions)
        rows_to_purge = self.game_look_up[-1][-1] - last_traj_act_len + 1 # one more row due to indexing
        print(f'{rows_to_purge} steps from the last trajectory have been purged due to missing data')
        if rows_to_purge > 0 : 
            self.game_look_up = self.game_look_up[:-rows_to_purge] # remove some idx from last traj that is not there
            self.priorities = self.priorities[:-rows_to_purge]
            print(f'total loaded steps: {len(self.game_look_up)}')

    def get_buffer(self):
        return self.buffer
        
    def save_files(self, save_buffer=False):
        dir = os.path.join(self.datapath, 'replay', str(self.replay_buffer_id))
        print('dir: ', dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        np.save(os.path.join(dir, 'prior.npy'), np.array(self.priorities))
        np.save(os.path.join(dir, 'game_look_up.npy'), np.array(self.game_look_up))
        np.save(os.path.join(dir, 'utils.npy'), np.array([self.base_idx, len(self.buffer)]))

        if save_buffer:
            for i, game in enumerate(self.buffer):
                path = os.path.join(dir, str(i))
                game.save_file(path)

    def save_pools(self, pools, gap_step):
        if self.make_dataset:
            buffer_size = self.size()
            print('Current size: ', buffer_size)

        for (game, priorities) in pools:
            # Only append end game
            # if end_tag:
            self.save_game(game, True, gap_step, priorities)

    def save_game(self, game, end_tag, gap_steps, priorities=None):
        if self.get_total_len() >= self.config.total_transitions:
            return

        if end_tag:
            self._eps_collected += 1
            valid_len = len(game)
        else:
            valid_len = len(game) - gap_steps

        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(game))]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            priorities = priorities.copy().reshape(-1)
            # priorities[valid_len:len(game)] = 0.
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]

        total = self.get_total_len()
        beg_index = max(total - self.tail_len, 0)
        self.tail_index += [idx for idx in range(beg_index, total)]

    def get_game(self, idx):
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game

    def prepare_batch_context(self, batch_size, beta):
        assert beta > 0

        total = self.get_total_len()

        # uniform
        if random.random() < self.config.uniform_ratio:
            _alpha = 0.
        else:
            _alpha = self._alpha

        probs = self.priorities ** _alpha

        # if total * self.keep_ratio >= batch_size + self.config.max_moves:
        #     begin_index = int(total * (1 - self.keep_ratio)) + self.config.max_moves
        #     probs[:begin_index] = 0.

        probs /= probs.sum()
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)

        # if self.tail_ratio > 0:
        #     tail_num = min(len(self.tail_index), int(self.tail_ratio * batch_size))
        #     if tail_num > 0:
        #         tail_lst = np.random.choice(self.tail_index, tail_num, replace=False)
        #         indices_lst[:tail_num] = tail_lst

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
        return context

    def update_games(self, game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times):
        for (game_idx, current_index_lst, distributions_lst, value_lst, make_time) in zip(game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times):
            self.update_game(game_idx, current_index_lst, distributions_lst, value_lst, make_time)

    def update_game(self, game_idx, current_index_lst, distributions_lst, value_lst, make_time):
        for i in range(len(make_time)):
            if make_time[i] > self.clear_time:
                game_id, game_pos = self.game_look_up[game_idx]
                game_id -= self.base_idx
                game = self.buffer[game_id]
                game.store_search_stats(distributions_lst[i], value_lst[i], current_index_lst[i], set_flag=True)

    def update_priorities(self, batch_indices, batch_priorities, make_time):
        for i in range(len(batch_indices)):
            if make_time[i] > self.clear_time:
                idx, prio = batch_indices[i], batch_priorities[i]
                self.priorities[idx] = prio

    def remove_to_fit(self):
        current_size = self.size()
        # print('Remove fit part, current size: ', current_size)
        # if current_size > self.soft_capacity:
        #     num_excess_games = current_size - self.soft_capacity
        #     self._remove(num_excess_games)
        # else:
            # num_excess_games = 100
            # while self.get_total_len() > self.transition_top:
            #     self._remove(num_excess_games)
            #     num_excess_games *= 2
        total_transition = self.get_total_len()
        if total_transition > self.transition_top:
            index = 0
            for i in range(current_size):
                total_transition -= len(self.buffer[i])
                if total_transition <= self.transition_top * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self.config.batch_size:
                self._remove(index + 1)

    def _remove(self, num_excess_games):
        excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        del self.game_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def clear_buffer(self):
        del self.buffer[:]

    def size(self):
        return len(self.buffer)

    def episodes_collected(self):
        return self._eps_collected

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_total_len(self):
        return len(self.priorities)

    def get_buffer_name(self):
        return self.buffer_name

    def over(self):
        if self.make_dataset:
            return self.get_total_len() >= self.transition_top
        return False