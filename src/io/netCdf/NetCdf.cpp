/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * IO-routines for reading / writing netCdf data.
 **/
#include <netcdf.h>
#include "NetCdf.h"

tsunami_lab::io::NetCdf::NetCdf(t_real i_dxy,
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
    int l_status = nc_create("output.nc", NC_CLOBBER, &m_ncidp);

    // define dimensions
    int l_dimXId, l_dimYId, l_dimTimeId;
    l_status = nc_def_dim(m_ncidp, "x", m_nx, &l_dimXId);
    l_status = nc_def_dim(m_ncidp, "y", m_ny, &l_dimYId);
    l_status = nc_def_dim(m_ncidp, "time", NC_UNLIMITED, &l_dimTimeId);

    // define variables
    int l_varXId, l_varYId, l_varTimeId, l_varHId, l_varHuId, l_varHvId, l_varBId;
    int l_dimB[2] = {l_dimXId, l_dimYId};
    int l_dimQ[3] = {l_dimTimeId, l_dimXId, l_dimYId};
    l_status = nc_def_var(m_ncidp, "x", NC_FLOAT, 1, &l_dimXId, &l_varXId);
    l_status = nc_def_var(m_ncidp, "y", NC_FLOAT, 1, &l_dimYId, &l_varYId);
    l_status = nc_def_var(m_ncidp, "time", NC_FLOAT, 1, &l_dimTimeId, &l_varTimeId);
    l_status = nc_def_var(m_ncidp, "height", NC_FLOAT, 3, l_dimQ, &l_varHId);
    l_status = nc_def_var(m_ncidp, "momentum_x", NC_FLOAT, 3, l_dimQ, &l_varHuId);
    l_status = nc_def_var(m_ncidp, "momentum_y", NC_FLOAT, 3, l_dimQ, &l_varHvId);
    l_status = nc_def_var(m_ncidp, "bathymetry", NC_FLOAT, 2, l_dimB, &l_varBId);

    // write attributes
    l_status = nc_put_var_float(m_ncidp, l_varBId, &i_b[0]); // TODO: Clean bathymetry of todo cells
}

void tsunami_lab::io::NetCdf::write(t_real const *i_h,
                                    t_real const *i_hu,
                                    t_real const *i_hv,
                                    std::ostream &io_stream)
{
}