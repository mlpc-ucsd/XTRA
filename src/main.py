# Code modified from EfficientZero

import argparse
import logging.config
import os
import subprocess
import json
from tkinter import N
import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir, set_seed
from core.args import get_args_parser as xtra_args
from config.atari import muzero_config, AtariConfig
import copy

ATARI_ACTION_SPACE = 18

def _log_test_score(test_score, test_path, summary_writer=None, total_steps=None):
    test_log = {
        'mean_score': test_score.mean(),
        'std_score': np.sqrt(test_score.var()),
    }

    # tensorboard logging
    if summary_writer and total_steps:
        for key, val in test_log.items():
            summary_writer.add_scalar('train/{}'.format(key), np.mean(val), total_steps)

    # console logging
    logging.getLogger('test').info('Test Mean Score: {}'.format(test_log['mean_score']))
    logging.getLogger('test').info('Test Std Score: {}'.format(test_log['std_score']))
    logging.getLogger('test').info('Saving video in path: {}'.format(test_path))

def _get_model_path(args, required=False):
    if required:
        assert args.load_model, "Model loading is needed for current operation"
        assert os.path.exists(args.model_path), "Model path doesn't exist"
        model_path = args.model_path
        print("INFO: Model loaded from {}".format(model_path))
    else:
        if args.load_model:
            assert os.path.exists(args.model_path), "Model path doesn't exist"
            model_path = args.model_path
            print("INFO: Model loaded from {}".format(model_path))
        else:
            model_path = None
            print("INFO: Model not loaded")
    return model_path

def _get_offline_data_path(tasks, path_prefix, sort_path=False, task_suffix='NoFrameskip-v4'):
    offline_data_dict = {}
    for task in tasks:
        task_full_name = task + task_suffix
        data_paths = glob(path_prefix + task_full_name + '/*')
        assert len(data_paths) > 0, "No data found for task {}".format(task)
        if sort_path:
            data_paths.sort(key=lambda x: \
            int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        offline_data_dict[task_full_name] = data_paths
    print('INFO: offline_data_list={}'.format(tasks))
    return offline_data_dict


def opr_train(args, muzero_config):
    summary_writer = SummaryWriter(args.exp_path, flush_secs=10)
    muzero_config.batch_size //= muzero_config.world_size
    print('INFO: batch_size={}'.format(muzero_config.batch_size))
    
    multitask = args.load_model
    if args.load_model:
        print("INFO: training from pretrained multi-task model")
        muzero_config.action_space_size = ATARI_ACTION_SPACE # max number of actions in Atari
    model_path = _get_model_path(args, required=False)

    if args.aux_offline:
        # load replay buffer path
        assert args.aux_data_list is not '', "aux_data_list is empty"                
        tasks, path_prefix = args.aux_data_list.split('/'), 'data/'
        aux_data_dict = _get_offline_data_path(tasks, path_prefix, sort_path=True)
        print('INFO: online_train= {} || online_test= {}'.format(args.env, args.env_test))
    else:
        aux_data_dict=None

    data_save_path = os.path.join(args.buffer_save_path, args.env)
    model, weights = train(muzero_config, summary_writer, model_path, 
                        game_info=args.game_info, multi_game=multitask, 
                        aux_data_dict=aux_data_dict, data_save_path=data_save_path)
    model.set_weights(weights)
    total_steps = muzero_config.training_steps + muzero_config.last_steps
    test_score, test_path = test(muzero_config, model.to('cuda'), total_steps, muzero_config.test_episodes,
                                    'cuda', render=False, save_video=False, final_test=True, game_info=args.game_info,
                                    eval_game=args.env_test, multi_game=multitask)
    _log_test_score(test_score, test_path, summary_writer, total_steps)

def opr_test(args, muzero_config):
    model_path = _get_model_path(args, required=True)
    weights_full, weights = torch.load(model_path), {}
    model = muzero_config.get_uniform_network().to('cuda')
    components_dir = {'R': 'representation', 'D': 'dynamics', 'P': 'prediction', 'S': 'projection'}
    for _, component in components_dir.items():
        weights.update({k: v for k, v in weights_full.items() if component in k})
    model.load_state_dict(weights, strict=False)

    # test
    config_test = copy.deepcopy(muzero_config)
    config_test.set_game(muzero_config.env_test, set_action=False)  
    multi_game = muzero_config.action_space_size == 18
    test_score, test_path = test(config_test, model, 0, args.test_episodes, device='cuda', render=False,
                                    save_video=False, use_imageio=False, final_test=True,
                                    game_info=args.game_info, eval_game=args.env_test, multi_game=multi_game)
    _log_test_score(test_score, test_path)

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser('XTRA training and evaluation script', parents=[xtra_args()])

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    
    # seeding random iterators
    set_seed(args.seed)

    # set config as per arguments
    args.exp_path, args.buffer_save_path = muzero_config.set_config(args)

    if args.opr == 'test':
        args.exp_path = args.exp_path.split('/')
        args.exp_path[1] = 'eval'
        args.exp_path = '/'.join(args.exp_path)
        muzero_config.exp_path = args.exp_path
    else:
        os.makedirs(args.buffer_save_path, exist_ok=True)
    args.exp_path, log_base_path = make_results_dir(args.exp_path, args)
    
    # save args to args.exp_path
    with open(os.path.join(args.exp_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # subprocess.run execute cp f in python
    subprocess.run(["cp", "-r", os.path.dirname(os.path.realpath(__file__)), args.exp_path])

    print('INFO: path: ', args.exp_path)

    # set-up logger
    init_logger(log_base_path)

    # load json config
    json_path = 'src_online_ft/config/atari/game_info.json'
    assert os.path.exists(json_path), "game information doesn't exist"
    with open(json_path) as f:
        args.game_info = json.load(f)

    try:
        # finetune a model from a multi-task pretrained model
        if args.opr == 'train':
            opr_train(args, muzero_config)
        # evaluate any model checkpoint
        if args.opr == 'test':
            opr_test(args, muzero_config)
        ray.shutdown()
        
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
