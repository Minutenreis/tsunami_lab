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
module load tools/python/3.8
module load compiler/gcc/11.2.0
python3.8 -m pip install --user scons
python3.8 -m pip install --user distro

# Enter your executable commands here
# Execute the compiled program
date
scons
./build/tsunami_lab -t 10 -u "Tsunami2d tohoku_gebco20_ucsb3_50m_displ.nc tohoku_gebco20_ucsb3_50m_bath.nc 600" -f 100 -o netcdf -k 5 50