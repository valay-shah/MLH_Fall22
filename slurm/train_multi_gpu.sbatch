#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4              
#SBATCH --cpus-per-task=4                
#SBATCH --time=30:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=mlh_baseline_multi_gpu
#SBATCH --output=slurm/jobs/baseline_multi_gpu.out
#SBATCH --error=slurm/jobs/baseline_multi_gpu.err

experiment_config=$1

cd /scratch/$USER/MLH_Fall22/;
mkdir -p slurm/jobs;

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/csp9835/penv/;
export PATH=/scratch/csp9835/penv/bin:$PATH;
srun python experiment.py --experiment $experiment_config
