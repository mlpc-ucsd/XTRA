import argparse

# credit: https://stackoverflow.com/a/64259328/12922880 
def float_range(mini, maxi, include_none=True):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument
         include_none - set to True if none is a valid value
    """
    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""
        if include_none and arg is None:
            return arg
        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f
    # Return function handle to checking function
    return float_range_checker

def get_args_parser():
    parser = argparse.ArgumentParser(description='XTRA Pytorch Implementation', add_help=False)
    
    ##############################################################################################
    # XTRA Training
    ##############################################################################################
    parser.add_argument('--env',  help='Name of the environment')
    parser.add_argument('--env_test',  help='Name of the test environment')
    parser.add_argument('--result_dir', default='results',  help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case',  choices=['atari'], default='atari', help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr',  choices=['train', 'test'])
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False, help='If enabled, logs additional values (gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False, help='Renders the environment (default: %(default)s)')
    parser.add_argument('--force', action='store_true', default=False, help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int, default=1, help='batch cpu actor')
    parser.add_argument('--gpu_actor', type=int, default=1, help='batch bpu actor')
    parser.add_argument('--p_mcts_num', type=int, default=64, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int, default=8, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=80, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float_range(0, 1, include_none=True), default=0.8, help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--amp_type',  choices=['torch_amp', 'none'],  help='choose automated mixed precision type')
    parser.add_argument('--use_priority', action='store_true', default=False, help='Uses priority for data sampling in replay buffer. Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=False, help='max priority')
    parser.add_argument('--use_latest_model', action='store_true', default=False, help='Use target model for bootstrap value estimation (default: %(default)s)')
    parser.add_argument('--write_back', action='store_true', default=False, help='write back')
    parser.add_argument('--target_moving_average', action='store_true', default=False, help='Use moving average target model or not')
    parser.add_argument('--test_episodes', type=int, default=32,  help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--reanalyze_part', type=str, default='paper', help='reanalyze part',  choices=['none', 'value', 'policy', 'paper', 'all'])
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+', choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'], help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none', help='debug string')
    parser.add_argument('--world_size', type=int, default=1, help='number of parallel training process')
    parser.add_argument('--use_detach', action='store_true', default=False, help='detach CPU batch maker and GPU maker')
    parser.add_argument('--opt_level', type=str, default='O1', help='opt level in amp')
    parser.add_argument('--num_simulations', type=int, default=50, help='number of mcts eval')
    parser.add_argument('--lr_init', type=float, default=0.2, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_min', type=float, default=5e-6, help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_warm_up', type=float, default=0.01, help='warmup')
    parser.add_argument('--aux_offline', action='store_true', default=False, help='aux model with offline data')
    parser.add_argument('--aux_data_list', type=str, default='', help='Name of the environment')
    parser.add_argument('--training_steps', type=int, default=100000, help='number of training steps')
    parser.add_argument('--lr_decay_steps', type=int, default=100000, help='number of training steps')
    parser.add_argument('--aux_data_decay', action='store_true', default=False, help='x')
    # new arguments
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='', help='load model path')

    parser.add_argument('--replay_start_idx', type=int, default=1, help='Add batch buffer')
    parser.add_argument('--replay_end_idx', type=int, default=1, help='Add batch buffer')
    parser.add_argument('--replay_data_idx', type=int, default=1, help='Add batch buffer')
    parser.add_argument('--if_mix_data', action='store_true', default=False, help='if_mix_data')

    # data synthesis
    parser.add_argument('--batch_size', type=int, default=256, help='batch bpu actor')

    parser.add_argument('--feature_mode', type=str, default='h_1024', choices=['s+sa_576','r+v_1208', 'r+v+sa_1208', 'h+sa_1024', 'r_608', 'v_608', 'h_1024', 'r+v+s_1784', 'r+v+h_2232'])
    parser.add_argument('--encoder_depth', default=2, type=int)
    parser.add_argument('--decoder_depth', default=2, type=int)

    parser.add_argument('--offline_data_list', type=str, default='', help='Name of the environment')
    parser.add_argument('--gray_scale', action='store_true', default=False)

    parser.add_argument('--optimizer', default='sgd', type=str, metavar='OPTIMIZER',  help='Optimizer (default: "sgd"')
    parser.add_argument('--lr_type', default='step', type=str, metavar='OPTIMIZER',  help='Optimizer (default: "step"')
    return parser