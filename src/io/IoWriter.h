/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Abstract base class for IO-writes.
 */
#ifndef TSUNAMI_LAB_IO_IOWRITER_H
#define TSUNAMI_LAB_IO_IOWRITER_H

#include "../constants.h"

namespace tsunami_lab
{
    namespace io
    {
        class IoWriter;
    }
}

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