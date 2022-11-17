#!/usr/bin/bash
#PBS -N hinton_repro_dsscdl
#PBS -q dssc_gpu
#PBS -l select=1:ncpus=24:mem=128g
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m bea
#PBS -M emanuele@ballarin.cc

cd "/u/dssc/s223459/Downloads/newhintorch/HinTorch"
source /u/dssc/s223459/bin/condalinks/activate HinTorch
jupyter nbconvert --execute --to notebook repro.ipynb
python -O run.py
exit 0
