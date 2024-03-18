#!/bin/bash
#SBATCH --job-name=eetu_alpha_train
#SBATCH --account=project_2009901
#SBATCH --partition=gpu
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --gres=gpu:v100:1


module load pytorch
srun python3 /users/eetusoro/project_2009901/seriaaliajot/pumpingironpower/Alpha/scr/train.py && python3 /users/eetusoro/project_2009901/seriaaliajot/pumpingironpower/Alpha/scr/predictions.py
