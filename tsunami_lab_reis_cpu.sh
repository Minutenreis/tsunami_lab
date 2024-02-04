#!/bin/bash
#SBATCH --job-name=tsunami_lab_reis_cpu
#SBATCH --output=tsunami_lab_reis_cpu.output
#SBATCH --error=tsunami_lab_reis_cpu.err
#SBATCH --partition=s_standard,s_hadoop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=72

# Load any necessary modules (if needed)
# module load mymodule
module load tools/python/3.8
module load compiler/gcc/11.2.0
module load nvidia/cuda/11.7
python3.8 -m pip install --user scons==4.0.1
python3.8 -m pip install --user distro

# Enter your executable commands here
# Execute the compiled program
date
cd /beegfs/gi24ken/tsunami_lab
scons cxxO=-Ofast
./build/tsunami_lab -t 1 -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" 4000