#!/bin/bash
#SBATCH --job-name=tsunami_lab_reis
#SBATCH --output=tsunami_lab_reis.output
#SBATCH --error=tsunami_lab_reis.err
#SBATCH --partition=s_hadoop,s_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=72

# Load any necessary modules (if needed)
# module load mymodule
# todo: load netcdf
# todo: checkout repo on ara!
# todo: include submodules!

# Enter your executable commands here
# Execute the compiled program
date
scons
#todo: add paths to files (use tohoku)
./build/tsunami_lab -t 10 -u "Tsunami2d <path_to_displ> <path_to_bath> 18000" -o netcdf 500