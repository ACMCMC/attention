#!/bin/bash
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=8
#SBATCH --mem=124gb                    # Job memory request
#SBATCH --time=19:45:00              # Time limit hrs:min:sec
#SBATCH --output=run_%j.log       # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aldan.creo@rai.usc.es
##SBATCH --qos=short
##SBATCH --begin=now+390minutes                     # Delay execution for 4 hours

# If we want reproducibility, we should use the same GPU
##SBATCH --gres=gpu:V100S:1
##SBATCH --gres=gpu:A100_40:1
##SBATCH --gres=gpu:A100_80:1
##SBATCH --gres=gpu

echo "Hostname: $(hostname)"
#module avail
ml CUDA/12.4
#module list
#export HF_DATASETS_CACHE="$HOME/.cache"
#export TRANSFORMERS_CACHE="$HOME/.cache"
export DATASETS_VERBOSITY=info
export EVALUATE_VERBOSITY=info
export TRANSFORMERS_VERBOSITY=info

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

source $HOME/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate attention
echo "Conda info:"
conda info
echo "Python route: $(which python)"

#export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
python average_statistics_from_results.py
