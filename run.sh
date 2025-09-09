#!/bin/bash
#SBATCH -p polariton
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

python dynamics.py
# python model.py
