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
      return l_bin + getDisplacement(i_x, i_y);
    else
      return -m_delta + getDisplacement(i_x, i_y);
  }
  // max(bin, delta) + d
  if (l_bin > m_delta)
    return l_bin + getDisplacement(i_x, i_y);
  else
    return m_delta + getDisplacement(i_x, i_y);
}

// todo: implement
tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getDisplacement(t_real i_x,
                                                                         t_real i_y) const
{
  return i_x - i_y;
}

// TODO: implement
tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getBathymetryBin(t_real i_x, t_real i_y) const
{
  return i_x - i_y;
}