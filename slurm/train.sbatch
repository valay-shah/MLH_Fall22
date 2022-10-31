#!/bin/bash -e

#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=4                
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=train
#SBATCH --output=slurm/jobs/train.out
#SBATCH --error=slurm/jobs/train.err

experiment_config=$1

cd /scratch/$USER/MLH_Fall22

module purge;
source venv/bin/activate;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
python experiment.py --experiment $experiment_config
