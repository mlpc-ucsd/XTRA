import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from .mcts import MCTS
import core.ctree.cytree as cytree
from .utils import select_action, prepare_observation_lst, set_seed, prepare_observation, str_to_arr
from .game import GameHistory
import imageio
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from torch.cuda.amp import autocast as autocast

def test(config, model, counter, test_episodes, device, render, save_video=False, final_test=False, use_imageio=False, game_info=None, eval_game=None, multi_game=False):
    import warnings
    # ignore warning
    warnings.filterwarnings('ignore')

    model.to(device)
    model.eval()

    game_name = eval_game
    actions_num = game_info[game_name]['total_action']
    actions_id  = game_info[game_name]['action_id']
    action_meaning={0:'NOOP',1:'FIRE',2:'UP',3:'RIGHT',4:'LEFT',5:'DOWN',6:'UPRIGHT',7:'UPLEFT',8:'DOWNRIGHT',\
                    9:'DOWNLEFT',10:'UPFIRE',11:'RIGHTFIRE',12:'LEFTFIRE',13:'DOWNFIRE',14:'UPRIGHTFIRE',15:'UPLEFTFIRE',\
                    16:'DOWNRIGHTFIRE',17:'DOWNLEFTFIRE'}

    action_id_dict = {}
    for idx, item in enumerate(actions_id):
        action_id_dict[idx] = item

    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))
    if use_imageio:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        video_writer = imageio.get_writer(os.path.join(save_path, 'video.mp4'), fps=20)

    time_list = []
    with torch.no_grad():
        envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                              video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(envs[_].env.action_space, max_length=config.max_moves, config=config) for
            _ in
            range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

        step = 0
        ep_ori_rewards = np.zeros(test_episodes)
        ep_clip_rewards = np.zeros(test_episodes)
        while not dones.all():
            if render:
                for i in range(test_episodes):
                    envs[i].render()

            if config.image_based:
                stack_obs = []
                for game_history in game_histories:
                    stack_obs.append(game_history.step_obs())
                stack_obs = prepare_observation_lst(stack_obs)
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

            # get time
            time_start = time.time()

            # MCTS
            with autocast():
                network_output = model.initial_inference(stack_obs.float(), actions_id)
            hidden_state_roots = network_output.hidden_state
            reward_hidden_roots = network_output.reward_hidden
            reward_sum_pool = network_output.reward_sum
            policy_logits_pool = network_output.policy_logits.tolist()

            roots = cytree.Roots(test_episodes, actions_num, config.num_simulations)
            roots.prepare_no_noise(reward_sum_pool, policy_logits_pool)
            MCTS(config).run_multi(roots, model, hidden_state_roots, reward_hidden_roots, actions_id, multi_game=multi_game)
            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()

            # get time
            time_end = time.time()
            time_ns = time_end - time_start
            time_list.append(time_ns)
            # get mean of time_list
            mean_time = np.mean(time_list)
            
            
            for i in range(test_episodes):
                if dones[i]:
                    continue

                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                action, _ = select_action(distributions, temperature=1, deterministic=True)

                obs, ori_reward, done, info = env.step(action)
                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward
                

                if use_imageio and i == 0:
                    video_writer.append_data(str_to_arr(obs))

            # get action meaning
            meaning = action_meaning[action_id_dict[action.item()]]
            print('{}: [{}]step {}--action={}, distribution={} [{}]'.format('MCTS', eval_game[:-len('NoFrameskip-v4')], step, action.item(), distributions,meaning))
            step += 1
            print('Mean score: {:5f}({:5f})  Time {}'.format(ep_clip_rewards.mean(), ep_ori_rewards.mean(), mean_time))
        env.close()

        if use_imageio:
            video_writer.close()

    return ep_ori_rewards, save_path
