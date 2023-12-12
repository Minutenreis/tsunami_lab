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

We already ran Chapter 6 50/50 on Ara and on our home PC's and didn't see any difference in output quality.

8.1.3 Compare time consumption to your local pc and add a timer for each steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All configurations were run with `./build/tsunami_lab -i -t 10 -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" 4000`.
(Tohoku, 4000m Cellsize, no FileIO)

Methodology: We ran the program 3 times and took the average of the 3 runs.

Ara: Skylake Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz

..
    todo: redo all tests with above methodology (also make it run icpc)
+---------------------------------------------------+-----------------------------+
| Configuration                                     | Time per Cell and Iteration |
+===================================================+=============================+
| g++ -Ofast -g                                     | 48ns                        |
+---------------------------------------------------+-----------------------------+
| g++ -O3 -g                                        | 51ns                        |
+---------------------------------------------------+-----------------------------+
| g++ -O2 -g                                        | 54ns                        |
+---------------------------------------------------+-----------------------------+
| g++ -O1 -g                                        | 73ns                        |
+---------------------------------------------------+-----------------------------+
| g++ -O0 -g                                        | 173ns                       |
+---------------------------------------------------+-----------------------------+
| ipcp -Ofast -g                                    |                             |
+---------------------------------------------------+-----------------------------+
| ipcp -O3 -g                                       |                             |
+---------------------------------------------------+-----------------------------+
| ipcp -O2 -g                                       |                             |
+---------------------------------------------------+-----------------------------+
| ipcp -O1 -g                                       |                             |
+---------------------------------------------------+-----------------------------+
| ipcp -O0 -g                                       |                             |
+---------------------------------------------------+-----------------------------+

Home-PC Justus Dreßler: Coffee Lake Intel(R) Core(TM) i5-8600K CPU @ 3.60GHz
Disclaimer: We commented out code involving :code:`std::filesystem::directory_iterator` because icpc seems to not handle its library.
The code shouldn't be touched in the runtime (disabled fileIO) but it may result in a smaller binary.


+-----------------------------------------------------+-----------------------------+
| Configuration                                       | Time per Cell and Iteration |
+=====================================================+=============================+
| g++ -Ofast -g -march=native -mtune=native           | 39.3ns                      |
+-----------------------------------------------------+-----------------------------+
| ipcp -Ofast -g -march=native -mtune=coffeelake      | 41.3ns                      |
+-----------------------------------------------------+-----------------------------+
| ipcp -Ofast -g -march=native -mtune=coffeelake -ipo | 41.7ns                      |
+-----------------------------------------------------+-----------------------------+
| g++ -O3 -g -march=native -mtune=native              | 43.3ns                      |
+-----------------------------------------------------+-----------------------------+
| g++ -Ofast -g                                       | 44.3ns                      |
+-----------------------------------------------------+-----------------------------+
| ipcp -O3 -g                                         | 44.7ns                      |
+-----------------------------------------------------+-----------------------------+
| ipcp -Ofast -g                                      | 45.0ns                      |
+-----------------------------------------------------+-----------------------------+
| ipcp -O2 -g                                         | 47.3ns                      |
+-----------------------------------------------------+-----------------------------+
| g++ -O3 -g                                          | 47.3ns                      |
+-----------------------------------------------------+-----------------------------+
| g++ -O2 -g                                          | 49.0ns                      |
+-----------------------------------------------------+-----------------------------+
| g++ -O1 -g                                          | 66.3ns                      |
+-----------------------------------------------------+-----------------------------+
| ipcp -O1 -g                                         | 72.7ns                      |
+-----------------------------------------------------+-----------------------------+
| g++ -O0 -g                                          | 153.0ns                     |
+-----------------------------------------------------+-----------------------------+
| ipcp -O0 -g                                         | 208.3ns                     |
+-----------------------------------------------------+-----------------------------+

..
    todo test ara

Ara seems to be roughly 5-10% faster than Justus Dreßler's home PC i5-8600K.

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