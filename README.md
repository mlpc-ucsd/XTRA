# On the Feasibility of Cross-Task Transfer with Model-Based Reinforcement Learning

Thank you for your interest in this project! This repository contains the official PyTorch implementation of **XTRA** from 

[On the Feasibility of Cross-Task Transfer with Model-Based Reinforcement Learning](https://arxiv.org/abs/2210.10763) by

[Yifan Xu](https://yfxu.com/)\*, [Nicklas Hansen](https://nicklashansen.github.io/)\*, [Zirui Wang](https://zwcolin.github.io), [Yung-Chieh Chan](https://www.linkedin.com/in/jerry-chan-yc/), [Hao Su](http://ai.ucsd.edu/~haosu/) and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/)

\* Equal Contribution

<p align="center">
  <br><img src='media/method.png?raw=true' width="800"/><br>
   <a href="https://arxiv.org/pdf/2210.10763.pdf">[Paper]</a>&emsp;<a href="https://nicklashansen.github.io/xtra">[Website]</a>
</p>

## Environment Setup
We provide scripts, data, and model checkpoints to run online finetuning on Atari100k benchmark with cross-tasks (i.e., using offline data).

1. Download the offline data for cross-tasks and pretrained model checkpoints. Link is [here](https://drive.google.com/drive/folders/1UcY9nnldmHYUizUi08C-fsM-tHAZRutW?usp=sharing). Make sure you have ~100G of storage available to store the data and model checkpoints. Once downloading is finished, move the zip files into the root directory of this repo and directly unzip these files. You should have folder `/data` and `/pretrained_ckpts` under this root directory.  

2. Install and use gcc-7 and g++-7 to compile C files. (if your system is already set up to use them, you can safely ignore this step):
```
sudo apt -y install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
```

3. Prepare the C packages for MCTS:
```
cd src/core/ctree && bash make.sh
```

4. Create a new conda environment by using the following code block (for full reproducibility, we used Anaconda 2022.05 version, which can be donwloaded using `wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh`:
```
conda create -n xtra python=3.7
conda activate xtra
pip install -r requirements.txt
```

## Running the XTRA Framework
We provide a sample script to run cross-task finetuning on `Alien` with cross-tasks `Amidar`, `BankHeist`, `MsPacman` and `WizardOfWor` from a offline multitask pretrained model from these cross-tasks. To run the framework, execute the following command (at the root directory of the repo, using the specified setup):
```
bash script xtra_finetune.sh
```

## Citation
```
@inproceedings{
xu2023on,
title={On the Feasibility of Cross-Task Transfer with Model-Based Reinforcement Learning},
author={Yifan Xu and Nicklas Hansen and Zirui Wang and Yung-Chieh Chan and Hao Su and Zhuowen Tu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=KB1sc5pNKFv}
}
```

Note: this codebase is based on an older version of EfficientZero implementation.
