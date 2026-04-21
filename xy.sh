#!/bin/bash
#
# Propagate environment variables to the compute node
#SBATCH --export=ALL
# Run in the standard partition (queue)
#SBATCH --partition=teaching
# Specify project account
#SBATCH --account=teaching
# Distribute processes in round-robin fashion, probs cap
#SBATCH --distribution=block:block
# No of cores required (max. of 16, 4GB RAM per core)
#SBATCH --ntasks=16
# Runtime (hard, HH:MM:SS)
#SBATCH --time=48:00:00
# Job name
#SBATCH --job-name=XY_metroplolis_new
# Output file
#SBATCH --output=xy64new-slurm-%j.out
# Modify the line below to run your program:

# Run properly

module load mpi

perf stat -e cycles,instructions,cache-misses mpirun -np 16 ./run_xymod.py --size 16 --n-temps 25 --n-equil 4000 --n-samples 5000 --sample-interval 2 --outfile xy_L16.npz
perf stat -e cycles,instructions,cache-misses mpirun -np 16 ./run_xymod.py --size 32 --n-temps 25 --n-equil 6000 --n-samples 5000 --sample-interval 2 --outfile xy_L32.npz
perf stat -e cycles,instructions,cache-misses mpirun -np 16 ./run_xymod.py --size 64 --n-temps 25 --n-equil 12000 --n-samples 5000 --sample-interval 2 --outfile xy_L64.npz
