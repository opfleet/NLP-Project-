#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=nlp-project # Job name
#SBATCH --output=nlp-project-%j.out
# Standard output and error log
#SBATCH --mail-user=ofleet@middlebury.edu
# Where to send mail
#SBATCH --mail-type=ALL
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=10gb
# Job memory request 
#SBATCH --partition=gpu-standard
# Partition (queue) 
#SBATCH --time=10:00:00
# Time limit hrs:min:sec 
#SBATCH --gres=gpu:2

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 
# Running your code here 
python nlp-project.py
# End of job info 
echo "Ending: "`date +"%D %T"`
