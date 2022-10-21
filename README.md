# Machine Learning for Healthcare

## Setting up the Virtual Environment
```bash
    python3 -m venv venv
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    pip install -r requirements.txt
```

If it is the first time using Weights & Biases, then `wandb login` should be executed before running an experiment.

## Running an Experiment
```bash
    source venv/bin/activate
    python experiment.py --experiment config/reference.yaml
```


### Running an Experiment on NYU HPC Greene with SLURM
First, clone the repository to `/scratch/$HOME`.
Then, execute the following:
```bash
    sbatch slurm/train.sbatch config/reference.yaml
```

## Experiment Configuration
This section will explain the configuration for running experiments.
All configurations are `.yaml` files and should be stored in the `config/` directory.

### Global Settings
These are at the top level of the file.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `name` | The experiment name which W&B runs are organized and the directory local model checkpoints are stored. | No Default |
| `seed` | The random seed for reproducibility. | `0` |
| `debug` | Indicates to use only 100 samples of all data splits for quick iteration. | `False` |

### W&B Settings
These are under the `wanb` keyword.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `save_dir` | Save directory for local run information. | `__pycache__/` |

### Dataset Settings
These are under the `dataset` keyword.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `name` | The dataset name. | No Default |
| `kwargs` | Keyword arguments for a PyTorch Lightning Data Module object such that nested key value pairs will be a dictionary. | No Default |


### Model Settings
These are under the `model` keyword.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `name` | The model name. | No Default |
| `dir_path` | The directory to store checkpoints of this model. | `checkpoints/` |
| `kwargs` | Keyword arguments for a PyTorch Lightning Module object such that nested key value pairs will be a dictionary. | No Default |

### Criterion Settings
These are under the `criterion` keyword which is in model settings.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `name` | The name of the criterion. | No Default |
| `kwargs` | Keyword arguments for a PyTorch `nn` loss function such that nested key value pairs will be a dictionary. | No Default |

### Train Settings
These are under the `train` keyword.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `max_epochs` | The maximum number of epochs to run for. | `1` |


### Optimizer Settings
These are under the `optimizer` keyword which is in train settings.

| **Keyword** | **Description** | **Default** |
| --- | --- | --- |
| `name` | The name of the optimizer. | No Default |
| `kwargs` | Keyword arguments for a PyTorch `optim` function such that nest key value pairs will be a dictionary. |