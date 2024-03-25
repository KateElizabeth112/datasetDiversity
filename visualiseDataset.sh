#!/bin/bash
#PBS -l walltime=3:00:00
#PBS -l select=1:ncpus=15:mem=60gb:ngpus=1:gpu_type=RTX6000
#PBS -N visualise_dataset

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

python3 visualiseDataset.py -d "TS" -l "false
