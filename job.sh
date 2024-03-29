#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J dlg
#BSUB -n 1
#BSUB -W 23:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/dlg_%J.out
#BSUB -e logs/dlg_%J.err

module load cuda/11.1
source .venv/bin/activate

which python3

echo "Running script..."
python3 run_experiment.py
