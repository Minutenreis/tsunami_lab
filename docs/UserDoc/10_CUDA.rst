10 CUDA
=======
Links
-----

`Github Repo <https://github.com/Minutenreis/tsunami_lab>`_

`User Doc <https://tsunami-lab.readthedocs.io/en/latest/>`_

Individual Contributions
------------------------

Justus Dreßler: all members contributed equally

Thorsten Kröhl: all members contributed equally

Julius Halank: all members contributed equally


Project Plan for CUDA Implementation
-------------------------------------

Introduction:
This project plan aims to implement Nvidia CUDA technology within a tight four-week
timeframe to accelerate Tsunami simulations. The plan outlines realistic goals, milestones,
work packages, and a schedule to efficiently execute the CUDA integration process.

Goals
^^^^^
1. Acceleration: Implement CUDA to significantly accelerate Tsunami simulations by a factor of at least 2 compared to multithreaded CPU processing.
    .. figure:: _static/performance.png
   :width: 700

   Our completely not made up expected performance improvement graph 
2. Compatibility: Ensure CUDA integration without compromising calculation accuracy or functionality.

Milestones with work packages and estimated time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Preparation Package (Week 1):*

* Literature Review:
  * Conduct a brief literature review to gain fundamental knowledge about CUDA
  * Summarize relevant information for the team
* Test Programs with CUDA Implementation:
  * Create small test programs to understand the basic concepts of CUDA
  * Document results and experiences for the team

*Framework Analysis and Modification Package (Week 1-2):*

* Identification of Key Areas:
  * Analyze the existing Tsunami simulation framework to identify key areas for CUDA integration
  * Document identified areas and their significance
* Codebase Modification and Initial Kernel Development:
  * Modify the codebase for smooth integration of CUDA parallelization
  * Initiate the development of initial CUDA kernels for identified simulation components
  * Document changes and initial kernel developments

*CUDA Implementation Package (Week 2-3):*

* Development of CUDA Kernels:
  * Develop CUDA kernels for individual simulation components based on identified key areas
  * Document kernel development and implemented functions
* Initial Tests for Performance Improvement:
  * Conduct initial tests to identify performance improvements resulting from CUDA implementation
  * Analyze test results and document observed enhancements

*Optimization and Testing Package (Week 3-4):*

* Optimization of CUDA Code:
  * Optimize the developed CUDA code to enhance efficiency
  * Utilize profiling tools such as NVPROF for detailed analysis and optimization
* Comprehensive Validation Tests:
  * Conduct comprehensive tests to validate calculation accuracy after CUDA implementation
  * Document test results and validate performance improvements compared to previous simulations

