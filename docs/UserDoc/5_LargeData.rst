Tsunami Report 5 Large Data Input & Output
=============================================

Links
-----

`Github Repo <https://github.com/Minutenreis/tsunami_lab>`_

`User Doc <https://tsunami-lab.readthedocs.io/en/latest/>`_

Individual Contributions
------------------------

Justus Dreßler: all members contributed equally

Thorsten Kröhl: all members contributed equally

Julius Halank: all members contributed equally

5.1. NetCDF Output
------------------

5.1.1 Provide a mechanism which shares netCDF's files with your solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We used the recommended library `libnetcdf-dev <https://packages.ubuntu.com/jammy/libnetcdf-dev>`_ to write the netCDF files.
To access the header file we supplied the path in scons with the following code:

.. code-block:: python

    # check for libs
    conf = Configure(env)
    if not conf.CheckLibWithHeader('netcdf','netcdf.h','c++'):
        print('Did not find netcdf.h, exiting!')
        Exit(1)   

Scons takes care of the linking process and we can use the library in our code.

5.1.2 Implement a class io::NetCdf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We implemented a class :code:`io::NetCdf` which is able to write the netCDF files.
To unify the methods we abstracted both io::NetCdf and io::Csv into a common interface io::IOWriter.

.. code-block:: cpp

    class tsunami_lab::io::IoWriter
    {
    public:
    /**
     * Virtual destructor for base class.
     **/
    virtual ~IoWriter(){};

    /**
     * @brief Initialize the writer.
     *
     * @param i_dxy cell size.
     * @param i_nx number of cells in x-direction.
     * @param i_ny number of cells in y-direction.
     * @param i_stride stride of the data.
     * @param i_ghostCellsX number of ghost cells in x-direction.
     * @param i_ghostCellsY number of ghost cells in y-direction.
     * @param i_offsetX offset in x-direction.
     * @param i_offsetY offset in y-direction.
     * @param i_b bathymetry.
     */
    virtual void init(t_real i_dxy,
                      t_idx i_nx,
                      t_idx i_ny,
                      t_idx i_stride,
                      t_idx i_ghostCellsX,
                      t_idx i_ghostCellsY,
                      t_real i_offsetX,
                      t_real i_offsetY,
                      t_real const *i_b) = 0;

    /**
     * @brief Writes the data to the output.
     *
     * @param i_h water height.
     * @param i_hu momentum in x-direction.
     * @param i_hv momentum in y-direction.
     * @param i_time time in simulation.
     * @param i_nOut number of the output.
     */
    virtual void write(t_real const *i_h,
                       t_real const *i_hu,
                       t_real const *i_hv,
                       t_real i_time,
                       t_idx i_nOut) = 0;
    };

    #endif

Meaning in the main method after setup is declared we no longer need to care which setup is called.
In the init method we initialize the netCDF file and write the dimensions and non time dependent variables.

.. code-block:: cpp

    void tsunami_lab::io::NetCdf::init(t_real i_dxy,
                                   t_idx i_nx,
                                   t_idx i_ny,
                                   t_idx i_stride,
                                   t_idx i_ghostCellsX,
                                   t_idx i_ghostCellsY,
                                   t_real i_offsetX,
                                   t_real i_offsetY,
                                   t_real const *i_b)
    {
    // saves setup parameters
    m_dxy = i_dxy;
    m_nx = i_nx;
    m_ny = i_ny;
    m_stride = i_stride;
    m_ghostCellsX = i_ghostCellsX;
    m_ghostCellsY = i_ghostCellsY;
    m_offsetX = i_offsetX;
    m_offsetY = i_offsetY;

    // create netCdf file
    ncCheck(nc_create("output.nc", NC_CLOBBER, &m_ncidp), __FILE__, __LINE__);

    // define dimensions & variables
    int l_dimXId, l_dimYId, l_dimTimeId;
    ncCheck(nc_def_dim(m_ncidp, "x", m_nx, &l_dimXId), __FILE__, __LINE__);
    ncCheck(nc_def_dim(m_ncidp, "y", m_ny, &l_dimYId), __FILE__, __LINE__);
    ncCheck(nc_def_dim(m_ncidp, "time", NC_UNLIMITED, &l_dimTimeId), __FILE__, __LINE__);

    int l_dimB[2] = {l_dimYId, l_dimXId};
    int l_dimQ[3] = {l_dimTimeId, l_dimYId, l_dimXId};
    ncCheck(nc_def_var(m_ncidp, "x", NC_FLOAT, 1, &l_dimXId, &m_varXId), __FILE__, __LINE__);
    ncCheck(nc_put_att_text(m_ncidp, l_dimXId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);
    ncCheck(nc_def_var(m_ncidp, "y", NC_FLOAT, 1, &l_dimYId, &m_varYId), __FILE__, __LINE__);
    ncCheck(nc_put_att_text(m_ncidp, l_dimYId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);
    ncCheck(nc_def_var(m_ncidp, "time", NC_FLOAT, 1, &l_dimTimeId, &m_varTimeId), __FILE__, __LINE__);
    ncCheck(nc_put_att_text(m_ncidp, l_dimTimeId, "units", strlen("seconds since simulationstart"), "seconds since simulationstart"), __FILE__, __LINE__);

    ncCheck(nc_def_var(m_ncidp, "height", NC_FLOAT, 3, l_dimQ, &m_varHId), __FILE__, __LINE__);
    ncCheck(nc_put_att_text(m_ncidp, m_varHId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);
    ncCheck(nc_def_var(m_ncidp, "momentum_x", NC_FLOAT, 3, l_dimQ, &m_varHuId), __FILE__, __LINE__);
    ncCheck(nc_put_att_text(m_ncidp, m_varHuId, "units", strlen("newton second"), "newton second"), __FILE__, __LINE__);
    if (m_ny > 1)
    {
        ncCheck(nc_def_var(m_ncidp, "momentum_y", NC_FLOAT, 3, l_dimQ, &m_varHvId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(m_ncidp, m_varHvId, "units", strlen("newton second"), "newton second"), __FILE__, __LINE__);
    }
    ncCheck(nc_def_var(m_ncidp, "bathymetry", NC_FLOAT, 2, l_dimB, &m_varBId), __FILE__, __LINE__);
    ncCheck(nc_put_att_text(m_ncidp, m_varBId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);

    // write data
    ncCheck(nc_enddef(m_ncidp), __FILE__, __LINE__);

    // generate x and y dimensions
    t_real *l_x = new t_real[m_nx];
    t_real *l_y = new t_real[m_ny];
    for (t_idx l_ix = 0; l_ix < m_nx; l_ix++)
    {
        l_x[l_ix] = m_offsetX + (l_ix + 0.5) * m_dxy;
    }
    for (t_idx l_iy = 0; l_iy < m_ny; l_iy++)
    {
        l_y[l_iy] = m_offsetY + (l_iy + 0.5) * m_dxy;
    }
    ncCheck(nc_put_var_float(m_ncidp, m_varXId, l_x), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(m_ncidp, m_varYId, l_y), __FILE__, __LINE__);

    // write bathymetry
    t_real *l_bPruned = pruneGhostCells(i_b);
    ncCheck(nc_put_var_float(m_ncidp, m_varBId, l_bPruned), __FILE__, __LINE__);
    delete[] l_x;
    delete[] l_y;
    delete[] l_bPruned;
    }

Where ncCheck is a function checking if the netCDF function was successful.

.. code-block:: cpp

    void tsunami_lab::io::ncCheck(int i_status, const char *i_file, int i_line)
    {
    if (i_status != NC_NOERR)
    {
        std::cerr << "Error: " << nc_strerror(i_status) << " in " << i_file << " at line " << i_line << std::endl;
        exit(1);
    }
    }

And pruneGhostCells is a function which removes the ghost cells from the bathymetry and returns a new allocated array of the result.

.. code-block:: cpp

    tsunami_lab::t_real *tsunami_lab::io::NetCdf::pruneGhostCells(t_real const *i_data)
    {
    t_real *l_outData = new t_real[m_nx * m_ny];
    for (t_idx l_ix = 0; l_ix < m_nx; l_ix++)
        for (t_idx l_iy = 0; l_iy < m_ny; l_iy++)
        {
            l_outData[l_iy * m_ny + l_ix] = i_data[(l_iy + m_ghostCellsY) * m_stride + (l_ix + m_ghostCellsX)];
        }
    return l_outData;
    }

With all this preprocessing the write step is really simple, we just prune the ghost cells of the height and momenta and write them to the netCDF file at the correct timestep.

.. code-block:: cpp

    void tsunami_lab::io::NetCdf::write(t_real const *i_h,
                                    t_real const *i_hu,
                                    t_real const *i_hv,
                                    t_real i_time,
                                    t_idx i_nOut)
    {
    // write data
    t_real *l_hPruned = pruneGhostCells(i_h);
    t_real *l_huPruned = pruneGhostCells(i_hu);

    size_t l_startp[3] = {i_nOut, 0, 0};
    size_t l_countp[3] = {1, m_ny, m_nx};

    ncCheck(nc_put_vara_float(m_ncidp, m_varHId, l_startp, l_countp, l_hPruned), __FILE__, __LINE__);
    ncCheck(nc_put_vara_float(m_ncidp, m_varHuId, l_startp, l_countp, l_huPruned), __FILE__, __LINE__);
    ncCheck(nc_put_var1_float(m_ncidp, m_varTimeId, &i_nOut, &i_time), __FILE__, __LINE__);

    delete[] l_hPruned;
    delete[] l_huPruned;

    // write momentum_y only if ny > 1 (2D)
    if (m_ny > 1)
    {
        t_real *l_hvPruned = pruneGhostCells(i_hv);
        ncCheck(nc_put_vara_float(m_ncidp, m_varHvId, l_startp, l_countp, l_hvPruned), __FILE__, __LINE__);
        delete[] l_hvPruned;
    }
    }  


5.2. NetCDF Input
-----------------

5.2.1 Implement new class setups::ArtificialTsunami2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We implemented a new class :code:`setups::ArtificialTsunami2d` with the displacement hardcoded as required:

.. code-block:: cpp

    tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getDisplacement(t_real i_x,
                                                                              t_real i_y) const
    {
        if (i_x >= -500 && i_x <= 500 && i_y >= -500 && i_y <= 500)
        {
            return 5 * getF(i_x, i_y) * getG(i_x, i_y);
        }
        return 0;
    }

    tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getF(t_real i_x,
                                                                   t_real) const
    {
        return std::sin(((i_x / 500) + 1) * M_PI);
    }

    tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getG(t_real,
                                                                   t_real i_y) const
    {
        return -(i_y / 500) * (i_y / 500) + 1;
    }

5.2.2 Extend your class io:NetCdf of Ch. 5.1 with support for reading netCDF files 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We added a read function to our :code:`io::NetCdf` which reads a netCdf file with the variables x,y,z where z is dependent on y and x.

.. code:: cpp

    void tsunami_lab::io::NetCdf::read(char *i_fileName,
                                   t_idx *o_nx,
                                   t_idx *o_ny,
                                   t_real **o_x,
                                   t_real **o_y,
                                   t_real **o_z)
    {   
    int l_ncidp = -1;
    // open netCdf file
    ncCheck(nc_open(i_fileName, NC_NOWRITE, &l_ncidp), __FILE__, __LINE__);

    // read dimensions
    int l_dimXId, l_dimYId;
    ncCheck(nc_inq_dimid(l_ncidp, "x", &l_dimXId), __FILE__, __LINE__);
    ncCheck(nc_inq_dimid(l_ncidp, "y", &l_dimYId), __FILE__, __LINE__);

    ncCheck(nc_inq_dimlen(l_ncidp, l_dimXId, o_nx), __FILE__, __LINE__);
    ncCheck(nc_inq_dimlen(l_ncidp, l_dimYId, o_ny), __FILE__, __LINE__);

    // read variables
    int l_varXId, l_varYId, l_varZId;
    ncCheck(nc_inq_varid(l_ncidp, "x", &l_varXId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "y", &l_varYId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "z", &l_varZId), __FILE__, __LINE__);

    *o_x = new t_real[*o_nx];
    *o_y = new t_real[*o_ny];
    *o_z = new t_real[(*o_nx) * (*o_ny)];

    ncCheck(nc_get_var_float(l_ncidp, l_varXId, *o_x), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varYId, *o_y), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varZId, *o_z), __FILE__, __LINE__);

    // close netCdf file
    ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);
    }


5.2.3 Integrate the new class setups::TsunamiEvent2d into your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also added a new class :code:`setups::TsunamiEvent2d` which reads a netCdf file for displacement and bathymetry values (both given as x,y,z(x,y)).
In the constructor the netCdf files are read and we calculate the width and height of the domain in the bathymetry.
We also get the offset in x- and y-direction (the minimum x and y value).

.. code-block:: cpp

    tsunami_lab::setups::TsunamiEvent2d::TsunamiEvent2d(char *i_displacement,
                                                    char *i_bathymetry,
                                                    t_real *o_width,
                                                    t_real *o_height,
                                                    t_real *o_offsetX,
                                                    t_real *o_offsetY)
    {
    // read netCDF files
    io::NetCdf::read(i_displacement, &m_ndX, &m_ndY, &m_displacementX, &m_displacementY, &m_displacement);
    io::NetCdf::read(i_bathymetry, &m_nbX, &m_nbY, &m_bathymetryX, &m_bathymetryY, &m_bathymetry);

    // calculate width
    *o_width = m_bathymetryX[m_nbX - 1] - m_bathymetryX[0];
    *o_height = m_bathymetryY[m_nbY - 1] - m_bathymetryY[0];
    *o_offsetX = m_bathymetryX[0];
    *o_offsetY = m_bathymetryY[0];
    }

We get the displacement at a given point by finding the closest x and y value in the displacement netcdf file, if the given x and y are within the bounds of the file.

.. code:: cpp

    tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getDisplacement(t_real i_x,
                                                                         t_real i_y) const
    {
    t_idx l_x = 0;
    t_idx l_y = 0;
    // check if in bounds
    if (i_x < m_displacementX[0] || i_x > m_displacementX[m_ndX - 1] || i_y < m_displacementY[0] || i_y > m_displacementY[m_ndY - 1])
        return 0;

    // find closest x and y
    for (t_idx l_ix = 0; l_ix < m_ndX; l_ix++)
    {
        if (m_displacementX[l_ix] > i_x)
        {
            if (i_x - m_displacementX[l_ix - 1] < m_displacementX[l_ix] - i_x)
                l_x = l_ix - 1;
            else
                l_x = l_ix;
            break;
        }
    }
    for (t_idx l_iy = 0; l_iy < m_ndY; l_iy++)
    {
        if (m_displacementY[l_iy] > i_y)
        {
            if (i_y - m_displacementY[l_iy - 1] < m_displacementY[l_iy] - i_y)
                l_y = l_iy - 1;
            else
                l_y = l_iy;
            break;
        }
    }

    // return displacement
    return m_displacement[l_y * m_ndX + l_x];
    }

The bathymetry gets calculated analogue to the displacement.

5.2.4 Check the correctness of your file input-based class setups::TsunamiEvent2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. video:: _static/5_ArtificialTsunami2d_300s_400.mp4
  :width: 700
  :autoplay:
  :loop:
  :nocontrols:
  :muted:

The Artificial Tsunami Setup simulated over 5 Minutes (300s).

.. video:: _static/5_Tsunami2d_300s_400.mp4
  :width: 700
  :autoplay:
  :loop:
  :nocontrols:
  :muted:

The Tsunami Event with the artificial Tsunami setups data given as netCdf files simulated over 5 Minutes (300s).

Both Setups seem to work very similarly, but to confirm we also added a Testcase that just runs the dataFiles through :code:`TsunamiEvent2d` and compares them to :code:`ArtificialTsunami2d`.

.. code-block:: cpp

    // check if artificialTsunami and tsunamiEvent2d are the same for the artificial tsunami data
  for (tsunami_lab::t_real l_x = -4985; l_x < 4985; l_x += 50)
    for (tsunami_lab::t_real l_y = -4985; l_y < 4985; l_y += 50)
    {
      REQUIRE(l_tsunamiEvent2d->getBathymetry(l_x, l_y) == Approx(l_artificialTsunami2d->getBathymetry(l_x, l_y)));
      REQUIRE(l_tsunamiEvent2d->getHeight(l_x, l_y) == Approx(l_artificialTsunami2d->getHeight(l_x, l_y)));
      REQUIRE(l_tsunamiEvent2d->getMomentumX(l_x, l_y) == Approx(l_artificialTsunami2d->getMomentumX(l_x, l_y)));
      REQUIRE(l_tsunamiEvent2d->getMomentumY(l_x, l_y) == Approx(l_artificialTsunami2d->getMomentumY(l_x, l_y)));
    }


