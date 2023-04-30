import math
import random
import numpy as np
import torch
import core.ctree.cytree as tree
from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def run_multi(self, roots, model, hidden_state_roots, reward_hidden_roots, actions_id=None, multi_game=False):

        if multi_game:
            action_id_dict = {}
            for idx, item in enumerate(actions_id):
                action_id_dict[idx] = item

        with torch.no_grad():
            model.eval()

            num = roots.num
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            hidden_state_index_x = 0
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len

            # DrawNode.clear()
            # d_root = DrawNode(0)
            # draw_tree = DrawTree(d_root)
            for index_simulation in range(self.config.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                results = tree.ResultsWrapper(num)
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
                search_lens = results.get_search_len()

                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])
                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to('cuda').unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to('cuda').unsqueeze(0)

                # TODO:WARNING: the order of the action is very important
                if multi_game:
                    last_actions = [action_id_dict[idx] for idx in last_actions]
                last_actions = torch.from_numpy(np.asarray(last_actions)).to('cuda').unsqueeze(1).long()

                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions, actions_id)
                else:
                    network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions, actions_id)

                hidden_state_nodes = network_output.hidden_state
                reward_sum_pool = network_output.reward_sum.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                if horizons > 0:
                    reset_idx = (np.array(search_lens) % horizons == 0)
                    assert len(reset_idx) == num
                    reward_hidden_nodes[0][:, reset_idx, :] = 0
                    reward_hidden_nodes[1][:, reset_idx, :] = 0
                    is_reset_lst = reset_idx.astype(np.int32).tolist()
                else:
                    is_reset_lst = [0 for _ in range(num)]

                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                tree.multi_back_propagate(hidden_state_index_x, discount,
                                          reward_sum_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results, is_reset_lst)
                # print(roots.get_distributions())
                # print(roots.get_values())
            #     trajs = roots.get_trajectories()
            #     print(trajs[0])
            #     d_root.add_traj(trajs[0])
            #     draw_tree.build()
            #
            # import ipdb
            # ipdb.set_trace()
            # draw_tree.make_video()


def get_node_distribution(root):
    depth_lst = []
    visit_count_lst = []

    # bfs
    node_stack = [root]
    while len(node_stack) > 0:
        node = node_stack.pop()

        if node.is_leaf():
            depth_lst.append(node.depth)
            visit_count_lst.append(node.visit_count)

        for action, child in node.children.items():
            if child.visit_count > 0:
                node_stack.append(child)

    # print(depth_lst)
    # assert (np.array(depth_lst) > 0).all()
    return depth_lst, visit_count_lst