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
module load compiler/intel/2020-Update2
python3.8 -m pip install --user scons
python3.8 -m pip install --user distro

# Enter your executable commands here
# Execute the compiled program
date
cd /beegfs/gi24ken/tsunami_lab
scons comp=g++ cxxO=-Ofast
OMP_NUM_THREADS=1 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=2 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=4 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=8 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=12 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=16 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=20 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=24 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=28 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=32 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=36 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=40 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=44 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=48 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=52 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=56 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=60 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=64 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=68 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
OMP_NUM_THREADS=72 ./build/tsunami_lab -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" -i 4000
