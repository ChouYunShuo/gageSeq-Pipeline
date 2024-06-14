#!/bin/bash

# SLURM OPTIONS
#-------------------------

#Which partition to use (this will either be ma2-gpu or ma-cpu)
#SBATCH -p ma1

#Which specific node to use (both of our GPU nodes, compute-4-26 and compute-4-13, map to the ma2-gpu partition, so if you need to specify a node, use this command)

#SBATCH -w compute-0-37

#How many cpus to use (default 1)
#SBATCH -c 8


#How much memory to allocate for your job
#SBATCH --mem 32GB

#%J gives the job number
#SBATCH --output /work/magroup/yunshuoc/Higashi_Pipeline/gageseq_package/logs/log-%J.txt 


#time to allocate for the job
#SBATCH -t 2-00:00:0

#SBATCH --job-name mouse2_slice99

# RUN SCRIPT       
#-------------------------
rm /scratch/tmp-yunshuo/h5_output/mouse2_slice99_001.h5
rm -rf /scratch/tmp-yunshuo/temp_h5
source /home/yunshuoc/miniconda3/etc/profile.d/conda.sh
conda activate higashi-env

cd /work/magroup/yunshuoc/Higashi_Pipeline/gageseq_package/gageseq2cellscope
python core.py
