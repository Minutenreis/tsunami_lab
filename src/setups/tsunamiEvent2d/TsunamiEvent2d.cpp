/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional tsunami event problem.
 **/
#include "TsunamiEvent2d.h"
#include "../../io/netCdf/NetCdf.h"
#include <stdexcept>
#include <iostream>

tsunami_lab::setups::TsunamiEvent2d::TsunamiEvent2d(char *i_displacement,
                                                    char *i_bathymetry,
                                                    t_real *o_width,
                                                    t_real *o_height,
                                                    t_real *o_offsetX,
                                                    t_real *o_offsetY)
{
  // read netCDF files
  io::NetCdf::read(i_displacement, &m_ndX, &m_ndY, &m_displacementX, &m_displacementY, &m_displacement);
  io::NetCdf::read(i_bathymetry, &m_nbX, &m_nbY, &m_bathymetryX, &m_bathymetryY, &m_bathymetry);

  // calculate cell size
  m_stepBathX = m_bathymetryX[1] - m_bathymetryX[0];
  m_stepBathY = m_bathymetryY[1] - m_bathymetryY[0];
  m_stepDisplX = m_displacementX[1] - m_displacementX[0];
  m_stepDisplY = m_displacementY[1] - m_displacementY[0];

  // get offsets
  m_offsetBathX = m_bathymetryX[0];
  m_offsetBathY = m_bathymetryY[0];
  m_offsetDisplX = m_displacementX[0];
  m_offsetDisplY = m_displacementY[0];

  // calculate width
  *o_width = m_bathymetryX[m_nbX - 1] - m_bathymetryX[0];
  *o_height = m_bathymetryY[m_nbY - 1] - m_bathymetryY[0];
  *o_offsetX = m_bathymetryX[0];
  *o_offsetY = m_bathymetryY[0];
}

tsunami_lab::setups::TsunamiEvent2d::~TsunamiEvent2d()
{
  delete[] m_displacementX;
  delete[] m_displacementY;
  delete[] m_displacement;
  delete[] m_bathymetryX;
  delete[] m_bathymetryY;
  delete[] m_bathymetry;
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getHeight(t_real i_x,
                                                                   t_real i_y) const
{
  t_real l_bin = getBathymetryBin(i_x, i_y);
  if (l_bin < 0)
    return std::max(-l_bin, m_delta);
  else
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
    return std::min(l_bin, -m_delta) + getDisplacement(i_x, i_y);
  else
    return std::max(l_bin, m_delta) + getDisplacement(i_x, i_y);
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getDisplacement(t_real i_x,
                                                                         t_real i_y) const
{
  // calculate closest x and y
  double l_x = round((i_x - m_offsetDisplX) / m_stepDisplX);
  double l_y = round((i_y - m_offsetDisplY) / m_stepDisplY);

  // check if in bounds
  if (l_x < 0 || l_x >= m_ndX || l_y < 0 || l_y >= m_ndY)
    return 0;

  // convert to t_idx
  t_idx l_x_idx = l_x;
  t_idx l_y_idx = l_y;

  // return displacement
  return m_displacement[l_y_idx * m_ndX + l_x_idx];
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getBathymetryBin(t_real i_x, t_real i_y) const
{

  // calculate closest x and y
  double l_x = round((i_x - m_offsetBathX) / m_stepBathX);
  double l_y = round((i_y - m_offsetBathY) / m_stepBathY);

  // check if in bounds
  if (l_x < 0)
    l_x = 0;
  if (l_x >= m_nbX)
    l_x = m_nbX - 1;
  if (l_y < 0)
    l_y = 0;
  if (l_y >= m_nbY)
    l_y = m_nbY - 1;

  // convert to t_idx
  t_idx l_x_idx = l_x;
  t_idx l_y_idx = l_y;

  // return bathymetry
  return m_bathymetry[l_y_idx * m_nbX + l_x_idx];
}