#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general'
#SBATCH --qos=short # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=00:50:00 # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1 # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=1 # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to$
#SBATCH --mem=50G # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu in$
#SBATCH --mail-type=END # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err # Set name of error log. %j is the Slurm jobId
#SBATCH --gres=gpu:a40
# Measure GPU usage of your job (initialization)
# previous=$(/usr/bin/nvidia-smi --query-accountedapps='gpu_utilization,mem_utilization,max_memory_usage,time' --format$# /usr/bin/nvidia-smi # Check sbatch settings are working (it should show the GPU that you requested)

module use /opt/insy/modulefiles
module load cuda/12.4 cudnn/12-8.9.1.23

source ~/.venv/bin/activate

srun python /home/nfs/jluu/model_code/main.py --type base --source_dataset oai --filename OAI_base_both_balanced --epochs 60 --batch_size 32 --lr 0.000005 --source_ratio 0.5 --target_ratio 0.5 --labda 0.0
# /usr/bin/nvidia-smi --query-accountedapps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /u$