/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * IO-routines for reading / writing netCdf data.
 **/

#include "../../constants.h"
#ifndef TSUNAMI_LAB_IO_NETCDF
#define TSUNAMI_LAB_IO_NETCDF

namespace tsunami_lab
{
    namespace io
    {
        class NetCdf;
    }
}

class tsunami_lab::io::NetCdf
{
private:
    //! cell width in x- and y-direction.
    t_real m_dxy;

    //! number of cells in x-direction.
    t_idx m_nx;

    //! number of cells in y-direction.
    t_idx m_ny;

    //! stride of the data arrays.
    t_idx m_stride;

    //! number of ghost cells in x-direction.
    t_idx m_ghostCellsX;

    //! number of ghost cells in y-direction.
    t_idx m_ghostCellsY;

    //! offset in x-direction.
    t_real m_offsetX;

    //! offset in y-direction.
    t_real m_offsetY;

    //! netCdf file id.
    int m_ncidp;

public:
    /**
     * Constructor.
     *
     * @param i_dxy cell width in x- and y-direction.
     * @param i_nx number of cells in x-direction.
     * @param i_ny number of cells in y-direction.
     * @param i_stride stride of the data arrays in y-direction (x is assumed to be stride-1).
     * @param i_ghostCellsX number of ghost cells in x-direction.
     * @param i_ghostCellsY number of ghost cells in y-direction.
     * @param i_offsetX offset in x-direction.
     * @param i_offsetY offset in y-direction.
     * @param i_b bathymetry of the cells.
     **/
    NetCdf(t_real i_dxy,
           t_idx i_nx,
           t_idx i_ny,
           t_idx i_stride,
           t_idx i_ghostCellsX,
           t_idx i_ghostCellsY,
           t_real i_offsetX,
           t_real i_offsetY,
           t_real const *i_b);

    /**
     * Writes the data as CSV to the given stream.
     * @param i_h water height of the cells; optional: use nullptr if not required.
     * @param i_hu momentum in x-direction of the cells; optional: use nullptr if not required.
     * @param i_hv momentum in y-direction of the cells; optional: use nullptr if not required.
     * @param io_stream stream to which the netCdf-data is written.
     **/
    void write(t_real const *i_h,
               t_real const *i_hu,
               t_real const *i_hv,
               std::ostream &io_stream);
};

#endif