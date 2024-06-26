#!/bin/bash
#SBATCH -A dep # account
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/nlp2_g09/Project/GoLLIE_MED/notebooks # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH -o tmp/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e tmp/%x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Para pedir gráficas

#!/bin/bash
echo "Checking CUDA devices:"
nvidia-smi

echo "Running Python diagnostics:"
python -c 'import torch; print("CUDA Available:", torch.cuda.is_available()); print("CUDA Version:", torch.version.cuda)'

python /hhome/nlp2_g09/Project/GoLLIE_MED/notebooks/medicalentityextraction.py