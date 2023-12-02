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

# Enter your executable commands here
# Execute the compiled program
date
cd tsunami_lab
scons
./build/tsunami_lab -t 10 -u "Tsunami2d chile_gebco20_usgs_250m_displ.nc chile_gebco20_usgs_250m_bath.nc 18000" -o netcdf 500