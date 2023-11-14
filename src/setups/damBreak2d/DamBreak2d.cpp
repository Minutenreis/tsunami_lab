/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional dam break problem.
 **/
#include "DamBreak2d.h"

tsunami_lab::t_real tsunami_lab::setups::DamBreak2d::getHeight(t_real i_x,
                                                               t_real i_y) const
{
  if (std::sqrt(i_x * i_x + i_y * i_y) < 10)
    return 10;
  else
    return 5;
}

tsunami_lab::t_real tsunami_lab::setups::DamBreak2d::getMomentumX(t_real,
                                                                  t_real) const
{
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::DamBreak2d::getMomentumY(t_real,
                                                                  t_real) const
{
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::DamBreak2d::getBathymetry(t_real,
                                                                   t_real) const
{
  // if (i_x >= 2 && i_x <= 4 && i_y >= 2 && i_y <= 4)
  // {
  //   return -2;
  // }
  // return -5;
  return 0;
}