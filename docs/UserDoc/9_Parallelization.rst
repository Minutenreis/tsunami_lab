9 Parallelization
=================

Links
-----

`Github Repo <https://github.com/Minutenreis/tsunami_lab>`_

`User Doc <https://tsunami-lab.readthedocs.io/en/latest/>`_

Individual Contributions
------------------------

Justus Dreßler: all members contributed equally

Thorsten Kröhl: all members contributed equally

Julius Halank: all members contributed equally


9.1&9.3 Parallize Solver with OpenMP
------------------------------------

We parallized our solver with OpenMP Pragmas. We added them in the following parts of :code:`WavePropagation2d`:

.. code:: cpp

    // init new cell quantities
    #pragma omp parallel for simd
      for (t_idx l_cy = 1; l_cy < m_nCellsy + 1; l_cy++)
        for (t_idx l_cx = 1; l_cx < m_nCellsx + 1; l_cx++)
        {
          m_hTemp[getCoord(l_cx, l_cy)] = m_h[getCoord(l_cx, l_cy)];
          m_huvTemp[getCoord(l_cx, l_cy)] = m_hu[getCoord(l_cx, l_cy)];
        }

    // iterate over edges and update with Riemann solutions in x direction
    #pragma omp parallel for
      for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
        for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
        {
          // determine left and right cell-id
          t_idx l_ceL = getCoord(l_ex, l_ey);
          t_idx l_ceR = getCoord(l_ex + 1, l_ey);

          // compute net-updates
          t_real l_netUpdates[2][2];

          /*Solver*/

          // update the cells' quantities
          m_h[l_ceL] -= i_scaling * l_netUpdates[0][0];
          m_hu[l_ceL] -= i_scaling * l_netUpdates[0][1];

          m_h[l_ceR] -= i_scaling * l_netUpdates[1][0];
          m_hu[l_ceR] -= i_scaling * l_netUpdates[1][1];
        }

      setGhostCellsY();
          // iterate over edges and update with Riemann solutions in y direction
    #pragma omp parallel for
      for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
        for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
        {
          // determine top and bottom cell-id
          t_idx l_ceB = getCoord(l_ex, l_ey);
          t_idx l_ceT = getCoord(l_ex, l_ey + 1);

          // compute net-updates
          t_real l_netUpdates[2][2];

          /*Solver*/

          // update the cells' quantities
          m_h[l_ceB] -= i_scaling * l_netUpdates[0][0];
          m_hv[l_ceB] -= i_scaling * l_netUpdates[0][1];

          m_h[l_ceT] -= i_scaling * l_netUpdates[1][0];
          m_hv[l_ceT] -= i_scaling * l_netUpdates[1][1];
        }

We swapped the order of the second loop back so we could parallize the outer loop without inducing dependencies between threads.
We also found a significant performance drop when we parallized the inner loop.
Its probably caused by the massive overhead of creating and destroying threads for each outer loop iteration.
Parallizing the inner loop raised our time per cell and iteration from 5ns to over 50ns (worse than the not parallized code that runs at roughly 28ns).

9.2 Runtime Parallel Solver
---------------------------

To make the solutions comparable to `8 Optimization`_ we also used :code:`./build/tsunami_lab -i -u "Tsunami2d output/tohoku_gebco20_usgs_250m_displ.nc output/tohoku_gebco20_usgs_250m_bath.nc 18000" 4000` as config.

+------------------------------------+--------------------+----------+
| Threads                            | Time per Iteration | Speedup  |
+====================================+====================+==========+
| 1                                  | 32.1518ns          | 1        |
+------------------------------------+--------------------+----------+
| 2                                  | 22.319ns           | 1.44     |
+------------------------------------+--------------------+----------+
| 4                                  | 12.6621ns          | 2.54     |
+------------------------------------+--------------------+----------+
| 8                                  | 7.4102ns           | 4.34     |
+------------------------------------+--------------------+----------+
| 18                                 | 4.2277ns           | 7.61     |
+------------------------------------+--------------------+----------+
| 19                                 | 3.9498ns           | 8.14     |
+------------------------------------+--------------------+----------+
| 24                                 | 3.5812ns           | 8.98     |
+------------------------------------+--------------------+----------+
| 36                                 | 3.2944ns           | 9.76     |
+------------------------------------+--------------------+----------+
| 37                                 | 3.8388ns           | 8.38     |
+------------------------------------+--------------------+----------+
| 48                                 | 3.73346ns          | 8.61     |
+------------------------------------+--------------------+----------+
| 72                                 | 4.6691ns           | 6.89     |
+------------------------------------+--------------------+----------+

.. figure:: _static/9_speedup.png
    :width: 700

    Speedup of the parallel solver over number of threads averaged over 3 runs.

It seems to rise approximately linear until 16 threads and then only very slowly rises over the remaining threads.
Using all threads imposed an interesting drastic drop in performance, probably because the last thread was also responsible for other programs and unbalances our workload.
Our maximum seems to be at 34 Threads so 2 threads short of 1 thread per core.

9.4 Scheduling and Pinning Strategies
-------------------------------------
