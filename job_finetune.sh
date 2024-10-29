#!/bin/bash
#SBATCH --job-name=finetune-vg          # Job name
#SBATCH --mail-type=ALL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ptieu28@tamu.edu  # Where to send mail
#SBATCH --nodes=1                        # Use one node
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --gres=gpu:tesla:1               # Type and number of GPUs 
#SBATCH --partition=gpu-research                  # Partition/Queue to run in
#SBATCH --qos=olympus-research-gpu       # Set QOS to use
#SBATCH --time=96:00:00                  # Time limit hrs:min:sec - set to 1 hour
python3 -u fine-tuning.py