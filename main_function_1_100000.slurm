#!/bin/bash
#SBATCH --job-name=main_test
#SBATCH --mail-user="s2465124@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --array 124-149
module load Miniconda3/4.7.10
conda init bash
source ~/.bashrc
conda activate test_env
export MKL_THREADING_LAYER=GNU
python main.py $SLURM_ARRAY_TASK_ID 1 100000


