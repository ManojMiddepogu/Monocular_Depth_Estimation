#!/bin/bash
#SBATCH --job-name=window_3_tiny

#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=80GB

#SBATCH --time=24:00:00

#SBATCH --output=./window_3_tiny.out
#SBATCH --error=./window_3_tiny.err
#SBATCH --export=ALL

singularity exec --bind /scratch/ --nv --overlay /scratch/crg9968/cv/overlay-50G-10M.ext3:ro /scratch/crg9968/cv/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash  -c "
source /ext3/env.sh
cd /scratch/crg9968/cv/Monocular_Depth_Estimation/vadepthnet
conda activate depth_estimation

python3 train.py ../configs/window_3_tiny_arguments_train_nyu.txt
"
