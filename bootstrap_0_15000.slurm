#!/bin/bash
#SBATCH --job-name=bootstrap_test
#SBATCH --mail-user="s2465124@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --array 200-201
module load Miniconda3/4.7.10
conda init bash
source ~/.bashrc
conda activate test_env

python bootstrap.py $SLURM_ARRAY_TASK_ID 0 15000


