#!/bin/bash -e

#SBATCH --account=csci_ga_3033_083-2022fa          
#SBATCH --partition=n1c10m64-v100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=4                
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --output=slurm/jobs/train.out
#SBATCH --error=slurm/jobs/train.err

experiment_config=$1

cd /scratch/$USER/MLH_Fall22

module purge;
source venv/bin/activate;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
python experiment.py --experiment $experiment_config
