#!/bin/bash
#SBATCH --job-name=deep_learning
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks-per-node=1

# Load any necessary modules
module load singularity cuda

# Run your Singularity container
singularity run --nv mycontainer.sif