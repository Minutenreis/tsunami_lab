#!/bin/bash
#SBATCH --job-name=tsunami_lab_reis
#SBATCH --output=tsunami_lab_reis.output
#SBATCH --error=tsunami_lab_reis.err
#SBATCH --partition=gpu_v100,gpu_p100,gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --cpus-per-task=48

# Load any necessary modules (if needed)
# module load mymodule
module load tools/python/3.8
module load compiler/gcc/11.2.0
module load compiler/intel/2020-Update2
module load nvidia/cuda/11.7
python3.8 -m pip install --user scons==4.0.1
python3.8 -m pip install --user distro

# Enter your executable commands here
# Execute the compiled program
date
cd /beegfs/gi24ken/tsunami_lab
scons
./build/tsunami_lab -t 10 -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -c 4000