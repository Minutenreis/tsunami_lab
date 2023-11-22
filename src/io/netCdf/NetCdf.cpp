/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * IO-routines for reading / writing netCdf data.
 **/
#include <netcdf.h>
#include <ncCheck.h>
#include "NetCdf.h"

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
    netCDF::ncCheck(nc_create("output.nc", NC_CLOBBER, &m_ncidp), __FILE__, __LINE__);

    // define dimensions & variables
    int l_dimXId, l_dimYId, l_dimTimeId;
    netCDF::ncCheck(nc_def_dim(m_ncidp, "x", m_nx, &l_dimXId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_dim(m_ncidp, "y", m_ny, &l_dimYId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_dim(m_ncidp, "time", NC_UNLIMITED, &l_dimTimeId), __FILE__, __LINE__);

    int l_dimB[2] = {l_dimYId, l_dimXId};
    int l_dimQ[3] = {l_dimTimeId, l_dimYId, l_dimXId};
    netCDF::ncCheck(nc_def_var(m_ncidp, "x", NC_FLOAT, 1, &l_dimXId, &m_varXId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_var(m_ncidp, "y", NC_FLOAT, 1, &l_dimYId, &m_varYId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_var(m_ncidp, "time", NC_FLOAT, 1, &l_dimTimeId, &m_varTimeId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_var(m_ncidp, "height", NC_FLOAT, 3, l_dimQ, &m_varHId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_var(m_ncidp, "momentum_x", NC_FLOAT, 3, l_dimQ, &m_varHuId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_var(m_ncidp, "momentum_y", NC_FLOAT, 3, l_dimQ, &m_varHvId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_def_var(m_ncidp, "bathymetry", NC_FLOAT, 2, l_dimB, &m_varBId), __FILE__, __LINE__);

    // write data
    netCDF::ncCheck(nc_enddef(m_ncidp), __FILE__, __LINE__);

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
    netCDF::ncCheck(nc_put_var_float(m_ncidp, m_varXId, l_x), __FILE__, __LINE__);
    netCDF::ncCheck(nc_put_var_float(m_ncidp, m_varYId, l_y), __FILE__, __LINE__);

    // write bathymetry
    t_real *l_bPruned = pruneGhostCells(i_b);
    netCDF::ncCheck(nc_put_var_float(m_ncidp, m_varBId, l_bPruned), __FILE__, __LINE__);
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
    // write data
    t_real *l_hPruned = pruneGhostCells(i_h);
    t_real *l_huPruned = pruneGhostCells(i_hu);
    t_real *l_hvPruned = pruneGhostCells(i_hv);

    size_t l_startp[3] = {i_nOut, 0, 0};
    size_t l_countp[3] = {1, m_ny, m_nx};

    netCDF::ncCheck(nc_put_vara_float(m_ncidp, m_varHId, l_startp, l_countp, l_hPruned), __FILE__, __LINE__);
    netCDF::ncCheck(nc_put_vara_float(m_ncidp, m_varHuId, l_startp, l_countp, l_huPruned), __FILE__, __LINE__);
    netCDF::ncCheck(nc_put_vara_float(m_ncidp, m_varHvId, l_startp, l_countp, l_hvPruned), __FILE__, __LINE__);
    netCDF::ncCheck(nc_put_var1_float(m_ncidp, m_varTimeId, &i_nOut, &i_time), __FILE__, __LINE__);

    delete[] l_hPruned;
    delete[] l_huPruned;
    delete[] l_hvPruned;
}

tsunami_lab::io::NetCdf::~NetCdf()
{
    if (m_ncidp != -1) // if file is open
        netCDF::ncCheck(nc_close(m_ncidp), __FILE__, __LINE__);
}