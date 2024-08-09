#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --gres=gpu:v100l
#SBATCH --mem=187G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-24:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/experiment_output.out
module load StdEnv/2023
module load gcc/12.3
module load eccodes/2.31.0
module load openmpi/4.1.5
module load hdf5/1.14.2
module load netcdf/4.9.2
source /home/zgoussea/geospatial/bin/activate
mpirun -np 1 python experiment.py --month 5 --n_epochs_init 30 --n_epochs_retrain 10 --hidden_size 32 --n_conv 3 --input_timesteps 10 --output_timesteps 90 --mesh_size 4 --mesh_type heterogeneous --conv_type GCNConv --directory experiment§