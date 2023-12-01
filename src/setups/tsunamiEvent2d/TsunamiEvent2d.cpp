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

  if (std::isnan(l_bin))
    std::cout << "nan bathymetry " << i_x << i_y << std::endl;
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

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getDisplacement(t_real i_x,
                                                                         t_real i_y) const
{
  t_idx l_x = 0;
  t_idx l_y = 0;
  // check if in bounds
  if (i_x < m_displacementX[0] || i_x > m_displacementX[m_ndX - 1] || i_y < m_displacementY[0] || i_y > m_displacementY[m_ndY - 1])
    return 0;

  // find closest x and y
  for (t_idx l_ix = 0; l_ix < m_ndX; l_ix++)
  {
    if (m_displacementX[l_ix] > i_x)
    {
      if (i_x - m_displacementX[l_ix - 1] < m_displacementX[l_ix] - i_x)
        l_x = l_ix - 1;
      else
        l_x = l_ix;
      break;
    }
  }
  for (t_idx l_iy = 0; l_iy < m_ndY; l_iy++)
  {
    if (m_displacementY[l_iy] > i_y)
    {
      if (i_y - m_displacementY[l_iy - 1] < m_displacementY[l_iy] - i_y)
        l_y = l_iy - 1;
      else
        l_y = l_iy;
      break;
    }
  }

  // return displacement
  return m_displacement[l_y * m_ndX + l_x];
}

tsunami_lab::t_real tsunami_lab::setups::TsunamiEvent2d::getBathymetryBin(t_real i_x, t_real i_y) const
{
  t_idx l_x = 0;
  t_idx l_y = 0;

  // find closest x and y
  if (i_x <= m_bathymetryX[0])
    l_x = 0;
  else if (i_x >= m_bathymetryX[m_nbX - 1])
    l_x = m_nbX - 1;
  else
  {
    for (t_idx l_ix = 1; l_ix < m_nbX; l_ix++)
    {
      if (m_bathymetryX[l_ix] > i_x)
      {
        if (i_x - m_bathymetryX[l_ix - 1] < m_bathymetryX[l_ix] - i_x)
          l_x = l_ix - 1;
        else
          l_x = l_ix;
        break;
      }
    }
  }

  if (i_y <= m_bathymetryY[0])
    l_y = 0;
  else if (i_y >= m_bathymetryY[m_nbY - 1])
    l_y = m_nbY - 1;
  else
  {
    for (t_idx l_iy = 1; l_iy < m_nbY; l_iy++)
    {
      if (m_bathymetryY[l_iy] > i_y)
      {
        if (i_y - m_bathymetryY[l_iy - 1] < m_bathymetryY[l_iy] - i_y)
          l_y = l_iy - 1;
        else
          l_y = l_iy;
        break;
      }
    }
  }

  // return bathymetry
  return m_bathymetry[l_y * m_nbX + l_x];
}