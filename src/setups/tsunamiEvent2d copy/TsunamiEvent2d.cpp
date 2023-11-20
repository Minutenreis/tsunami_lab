/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional tsunami event problem.
 **/
#include "TsunamiEvent2d.h"

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getHeight(t_real i_x,
                                                               t_real i_y) const
{
  t_real l_bin = getBathymetryBin(i_x, i_y);
  if (l_bin < 0)
  {
    if (-l_bin > m_delta)
      return -l_bin;
    else
      return m_delta;
  }
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getMomentumX(t_real,
                                                                  t_real) const
{
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getMomentumY(t_real,
                                                                  t_real) const
{
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getBathymetry(t_real i_x,
                                                                   t_real i_y) const
{
  t_real l_bin = getBathymetryBin(i_x, i_y);
  if (l_bin < 0)
  {
    if (l_bin < -m_delta)
      return l_bin + getDisplacement(i_x);
    else
      return -m_delta + getDisplacement(i_x);
  }
  // max(bin, delta) + d
  if (l_bin > m_delta)
    return l_bin + getDisplacement(i_x);
  else
    return m_delta + getDisplacement(i_x);
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getDisplacement(t_real i_x,
                                                                              t_real i_y) const
{
  return 5*getF(i_x, i_y)*getG(i_x, i_y);
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getF(t_real i_x,
                                                                              t_real) const
{
  return std::sin(((i_x/500) + 1) * M_PI);
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getG(t_real,
                                                                              t_real i_y) const
{
  - (i_y/500) * (i_y/500) + 1;
}

//TODO: implement
tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getBathymetryBin(t_real i_x, t_real i_y) const
{
  return 0;
}