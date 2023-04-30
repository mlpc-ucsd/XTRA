export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# pretrained model ckpt
path='pretrained_ckpts/amidar+bankheist+mspacman+wizardofwor.p'

# target game
game_name='Alien'
game_name=$game_name'NoFrameskip-v4'

# cross-tasks
aux_games='Amidar/BankHeist/MsPacman/WizardOfWor' 

exp_name='Alien-Amidar_BankHeist_MsPacman_WizardOfWor'

seed=4

python src/main.py \
    --env $game_name \
    --case atari \
    --opr train \
    --seed $seed \
    --num_gpus 4 \
    --num_cpus 96 \
    --force \
    --cpu_actor 14 \
    --gpu_actor 20 \
    --use_detach \
    --p_mcts_num 8 \
    --use_priority \
    --use_max_priority \
    --revisit_policy_search_rate 1.0 \
    --amp_type 'torch_amp' \
    --reanalyze_part 'paper' \
    --test_episodes 32 \
    --batch_size 256 \
    --info $exp_name \
    --env_test $game_name \
    --lr_init 0.2 \
    --lr_type 'step' \
    --optimizer 'pcgrad' \
    --lr_warm_up 0.01 \
    --load_model \
    --model_path $path \
    --aux_offline \
    --aux_data_list $aux_games \
    --aux_data_decay
