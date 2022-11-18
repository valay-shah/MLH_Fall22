# Machine Learning for Healthcare

If it is the first time using Weights & Biases, then `wandb login` should be executed before running an experiment.

## Running an Experiment
```bash
    conda activate /scratch/csp9835/penv
    python experiment.py --experiment config/reference.yaml
```

### Running an Experiment on NYU HPC Greene with SLURM
First, clone the repository to `/scratch/$HOME`.
Then, execute the following:
```bash
    sbatch slurm/train.sbatch config/reference.yaml
```
where `slurm/train.sbatch` is the type of SLURM job and `config/reference.yaml` is the experiment YAML configuration.

## Experiment Configuration
This section will explain the configuration for running experiments.
All configurations are `.yaml` files and should be stored in the `config/` directory.

| **Keyword** | **Description** | **Default** | **Required** |
| --- | --- | --- | --- |
| `name` | The experiment name which W&B runs are organized and the directory local model checkpoints are stored. | No Default | Yes |
| `mode` | The type of experiment to run (i.e., pretrain, downstream, or evaluate). | No Default | Yes |
| `seed` | The random seed for reproducibility. | `0` | No |
| `debug` | Indicates to use only 100 samples of all data splits for quick iteration. | `False` | No |
| `batch_size` | Batch size for training, validation, and test data loaders. | `32` | No |
| `num_workers` | Number of workers for training, validation, and test data loaders. | `4` | No |
| `checkpoint_path` | Directory to put saved model. | `checkpoints/` | No |
| `model_kwargs` | Model kwargs (e.g., hyperparameters). | No Default | No |
| `criterion_kwargs` | Criterion kwargs (e.g., loss). | No Default | No |
| `optimizer_kwargs` | Optimizer kwargs (e.g., gradients). | No Default | No |
| `train` | Section containing training specific options. | No Default | No |
| `downstream` | Section containing downstream specific options. | No Default | No |
