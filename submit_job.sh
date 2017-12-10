#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH -A comsm0018
#SBATCH -t 0-02:00
#SBATCH --mem=4G

module load libs/tensorflow/1.2
rm -rf logs
mkdir logs 
python traffic-signs.py file to run	