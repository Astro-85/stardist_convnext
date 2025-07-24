#!/bin/bash

module purge
module load modules/2.3
module load python 
module load cuda cudnn nccl

source /mnt/home/alu10/envs/stardist/bin/activate

# Slurm command: sbatch -p gpu --nodes=1 --gpus=4 -C a100-80gb --cpus-per-gpu=4 run_training_4D.sh

python -u `which torchrun` \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 4 \
    train.py --yaml_conf ./confs/train_convnext_unet_base-4D.yaml
