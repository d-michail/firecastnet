#!/bin/bash

#SBATCH --job-name=firecastnet
#SBATCH --partition=obiwan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --dependency=singleton
#SBATCH --error=err-firecastnet.err
#SBATCH --output=out-firecastnet.out
#SBATCH --mem=80G

# Activate Anaconda work environment
. ~/miniconda3/etc/profile.d/conda.sh
conda activate graphcast

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

MODEL_NAME="FireCastNet"
CONFIG_FILE="configs/config-wandb.yaml"
TIME="24"
SHIFT="0"
NAME="seasfire-bench-ts-${TIME}-shift-${SHIFT}"
LOG_DIR="lightning_logs/${NAME}"
CKPT_FILE="lightning_logs/${NAME}/checkpoints/last.ckpt"

srun python main.py test --model "$MODEL_NAME" --config "$CONFIG_FILE" --trainer.logger.init_args.id="$NAME" --ckpt_path $1 --data.target_shift="$SHIFT" --model.init_args.timeseries_len="$TIME" --model.init_args.embed_cube_time="$TIME" --data.timeseries_weeks="$TIME"
