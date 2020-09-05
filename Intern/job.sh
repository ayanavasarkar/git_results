#!/bin/bash

#SBATCH --job-name=eff_net1

#SBATCH --gres=gpu:1

#SBATCH --mem=100gb

#SBATCH --output=eff_net1.out

#SBATCH --error=eff_net1.out

#SBATCH --partition=titanx-long

#SBATCH --nodes=1 

#!SBATCH --ntasks=4

#!SBATCH --ntasks-per-node=2

#SBATCH --time=04-23:00:00


module avail miniconda
source activate py3

python mura_new_data.py
