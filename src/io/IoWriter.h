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