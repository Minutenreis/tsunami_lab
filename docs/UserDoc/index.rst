.. Tsunami-Simulation documentation master file, created by
   sphinx-quickstart on Mon Oct 23 20:12:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :hidden:

   self
   1_RiemannSolver
   2_FiniteVolumeDiscretization
   3_BathymetryBoundaryConditions
   4_TwoDimensionalSolver
   5_LargeData
   6_TsunamiSimulations
   7_Checkpointing
   8_Optimization
   9_Parallelization
   10_CUDA
   10_1_CUDA
   doxygen

User Documentation
==================

Installing and Running
----------------------


* clone the project with :code:`git clone https://github.com/Minutenreis/tsunami_lab.git` 
* add the submodules with :code:`git submodule init` and :code:`git submodule update`
* install dependencies with :code:`apt-get install libnetcdf-dev`
* build with :code:`scons`
* execute the Program with :code:`./build/tsunami_lab [-s solver] [-u setup] [-b "boundary_left boundary_right"] [-r stationsJson] [-o outputType] [-f frames] [-t maxtime] [-k size] [-i] [-c] n_cells_x` 
* execute the tests with :code:`./build/tests`

The output of Solver is saved in :code:`/solutions` if you use :code:`csv` as outputType.
If you use :code:`netCdf` as outputType, the output is saved as :code:`output.nc` in the root directory.
The output of the Stations is in :code:`/stations`

Command Line Parameters
-----------------------

| :code:`n_cells_x` = number of cells the simulation gets broken up into in x-direction; y-direction depends on setup; specially cased in the tsunami2d Setup to mean the cell length instead.
| :code:`[-s solver]` = choose between :code:`roe` and :code:`fWave` solver, default is :code:`fWave`
| :code:`[-u setup]` = choose between :code:`'DamBreak1d h_l h_r'`, :code:`'ShockShock1d h hu'`,
 :code:`'RareRare1d h hu'`, :code:`'Custom1d h_l h_r hu_l hu_r middle'`, :code:`Subcrit1d`,
 :code:`Supercrit1d`, :code:`'Tsunami1d path_to_csv time_simulated'`, :code:`'ArtificialTsunami2d time_simulated'`,
 :code:`'Tsunami2d path_to_displacement path_to_bathymetry time_simulated'`, :code:`DamBreak2d` , default is :code:`DamBreak2d`
| :code:`[-b 'boundary_left boundary_right boundary_bottom boundary_top']` = 
 choose each boundary between :code:`wall` and :code:`open`, 
 default is :code:`open` for each; any boundary left out is set to :code:`open`
| :code:`[-r stationsJson]` = path of the stations json file, default is :code:`src/data/stations.json`
| :code:`[-o outputType]` = outputtype, choose between :code:`csv` and :code:`netCdf`, default is :code:`netCdf`
| :code:`[-f frames]` = (minimum) number of frames to be saved, default is :code:`100`
| :code:`[-t maxtime]` = maxTime of simulation, default is :code:`24`
| :code:`[-k size]` = size of cells in output as faktor, default is :code:`1`
| :code:`[-i]` = no fileIO (benchmarking)
| :code:`[-c]` = use CUDA (only for 2d)

:code:`stationsJson` Format:

.. code:: javascript

   {
      "period": float,
      "stations": [
         {
            "name": string,
            "x": float,
            "y": float
         },
         ...
      ]
   }

