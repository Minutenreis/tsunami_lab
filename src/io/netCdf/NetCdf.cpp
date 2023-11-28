/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * IO-routines for reading / writing netCdf data.
 **/
#include <netcdf.h>
#include "NetCdf.h"
#include <iostream>
#include <string.h>

void tsunami_lab::io::NetCdf::ncCheck(int i_status, char const *i_file, int i_line)
{
    if (i_status != NC_NOERR)
    {
        std::cerr << "NetCdf Error: " << nc_strerror(i_status) << " in " << i_file << " at line " << i_line << std::endl;
        exit(EXIT_FAILURE);
    }
}

tsunami_lab::t_real *tsunami_lab::io::NetCdf::pruneGhostCells(t_real const *i_data)
{
    t_real *l_outData = new t_real[m_nx * m_ny];
    for (t_idx l_ix = 0; l_ix < m_nx; l_ix++)
        for (t_idx l_iy = 0; l_iy < m_ny; l_iy++)
        {
            l_outData[l_iy * m_nx + l_ix] = i_data[(l_iy + m_ghostCellsY) * m_stride + (l_ix + m_ghostCellsX)];
        }
    return l_outData;
}

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

void tsunami_lab::io::NetCdf::write(t_real const *i_h,
                                    t_real const *i_hu,
                                    t_real const *i_hv,
                                    t_real i_time,
                                    t_idx i_nOut)
{
    if (m_ncidp == -1)
    {
        std::cerr << "NetCdf Error: File not initialized!" << std::endl;
        exit(EXIT_FAILURE);
    }
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

tsunami_lab::io::NetCdf::~NetCdf()
{
    if (m_ncidp != -1) // if file is open
        ncCheck(nc_close(m_ncidp), __FILE__, __LINE__);
}

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
