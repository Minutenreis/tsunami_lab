/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional artificial tsunami problem.
 **/
#include "ArtificialTsunami2d.h"

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getHeight(t_real i_x,
                                                                        t_real i_y) const
{
  return 100;
}

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getMomentumX(t_real,
                                                                           t_real) const
{
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getMomentumY(t_real,
                                                                           t_real) const
{
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getBathymetry(t_real i_x,
                                                                            t_real i_y) const
{
  t_real l_bin = -100;
  if (l_bin < 0)
  {
    // min(bin, -delta) + d
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

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getDisplacement(t_real i_x,
                                                                              t_real i_y) const
{
  if (i_x >= -500 && i_x <= 500 && i_y >= -500 && i_y <= 500)
  {
    return 5 * getF(i_x, i_y) * getG(i_x, i_y);
  }
  return 0;
}

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getF(t_real i_x,
                                                                   t_real) const
{
  return std::sin(((i_x / 500) + 1) * M_PI);
}

tsunami_lab::t_real tsunami_lab::setups::ArtificialTsunami2d::getG(t_real,
                                                                   t_real i_y) const
{
  return -(i_y / 500) * (i_y / 500) + 1;
}