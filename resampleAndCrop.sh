#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=15:mem=60gb
#PBS -N crop_AMOS

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

python3 resampleAndCrop.py -d "AMOS" -l "remote"