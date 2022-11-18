#!/bin/bash -e

#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=4                
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=mlh_test
#SBATCH --output=slurm/jobs/mlh_test.out
#SBATCH --error=slurm/jobs/mlh_test.err

experiment_config=$1

cd /scratch/$USER/MLH_Fall22/;
mkdir -p slurm/jobs;

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/csp9835/penv/;
export PATH=/scratch/csp9835/penv/bin:$PATH;
python experiment.py --experiment $experiment_config
