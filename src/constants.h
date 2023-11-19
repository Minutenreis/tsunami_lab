/**
 * @author Alexander Breuer (alex.breuer AT uni-jena.de)
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Constants / typedefs used throughout the code.
 **/
#ifndef TSUNAMI_LAB_CONSTANTS_H
#define TSUNAMI_LAB_CONSTANTS_H

#include <cstddef>
#include <string>

namespace tsunami_lab
{
  //! integral type for cell-ids, pointer arithmetic, etc.
  typedef std::size_t t_idx;

  //! floating point type
  typedef float t_real;

  //! boundary conditions
  enum class t_boundary
  {
    WALL,
    OPEN,
  };

  //! station data
  struct t_station
  {
    std::string name;
    t_real x;
    t_real y;
  };
}

#endif