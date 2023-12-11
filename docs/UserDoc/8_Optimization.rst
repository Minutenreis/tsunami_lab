Tsunami Report 8 Optimization
================================

Links
-----

`Github Repo <https://github.com/Minutenreis/tsunami_lab>`_

`User Doc <https://tsunami-lab.readthedocs.io/en/latest/>`_

Individual Contributions
------------------------

Justus Dreßler: all members contributed equally

Thorsten Kröhl: all members contributed equally

Julius Halank: all members contributed equally


8.1 ARA
-------------

8.1.1 Upload and Compile Code on Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.1.2 Run different scenarious and batch jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All configurations were run with `./build/tsunami_lab -i -t 10 -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" 4000`.
(Tohoku, 4000m Cellsize, no FileIO)


+-----------------------+-----------------------+-----------------------------+
| Configuration         | Time                  | Time per Cell and Iteration |
+=======================+=======================+=============================+
| g++ -O3 -g            | 42s 223ms 639us 759ns | 51ns                        |
+-----------------------+-----------------------+-----------------------------+
| g++ -O2 -g            | 42s 335ms 328us 612ns | 54ns                        |
+-----------------------+-----------------------+-----------------------------+
| g++ -O1 -g            | 56s 807ms 182us 673ns | 73ns                        |
+-----------------------+-----------------------+-----------------------------+
| g++ -O0 -g            | 1m 1s  1ms  1us  1ns  | 80ns                        |

8.1.3 Compare time consumption to your local pc and add a timer for each steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.2 Compiler
-------------

8.2.1 Add support for generic compilers to your build script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.2.2 Recompile your code using recent versions of the GNU and Intel compilers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.2.3 Compile your code using both compilers and try different optimization switches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.2.4 Make yourself familiar with optimization reports and add an option for them in your build script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.3 Instrumentation and Performance Counters
--------------------------------------------

8.3.1 Analyze your code with VTune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.3.2 Run the same analysis through the command line tool in a batch job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.3.3 Use the GUI to visualize the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.3.4 "Which parts are compute-intensive? Did you expect this?"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.3.5 Think about how you could improve the performance of your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8.3.6 (optional) Instrument your code manually using Score-P. Use Cube for the visualization of your measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^





8.2