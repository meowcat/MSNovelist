
# Training

* `run_train.sh`: run MSNovelist Singularity container and start `train.sh` in the container. For usage with SLURM, sets up one GPU. Use with `sbatch --array=0-9` to train 10-CV.
* `train.sh`: set environment variables and run `train.py`
* `train.py`: train one fold of the model

# Evaluation 

* `run_evaluation.sh`: run MSNovelist Singularity container and start `evaluation.sh` in the container. For usage with SLURM, no GPU. Use with `sbatch --array=0-9` to evaluate 10-CV.
* `evaluation.sh`: set environment variables and run `evaluation.py`
* `evaluation.py`: evaluate one fold of the model