/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional wave propagation patch.
 **/
#include "WavePropagation2d.h"
#include "../../solvers/roe/Roe.h"
#include "../../solvers/fWave/FWave.h"
#include <algorithm>

tsunami_lab::patches::WavePropagation2d::WavePropagation2d(t_idx i_nCellsx,
                                                           t_idx i_nCellsy,
                                                           bool i_useFWave,
                                                           t_boundary i_boundaryLeft,
                                                           t_boundary i_boundaryRight,
                                                           t_boundary i_boundaryBottom,
                                                           t_boundary i_boundaryTop) : m_nCellsx(i_nCellsx),
                                                                                       m_nCellsy(i_nCellsy),
                                                                                       m_useFWave(i_useFWave),
                                                                                       m_boundaryLeft(i_boundaryLeft),
                                                                                       m_boundaryRight(i_boundaryRight),
                                                                                       m_boundaryBottom(i_boundaryBottom),
                                                                                       m_boundaryTop(i_boundaryTop)
{

  // allocate memory including a single ghost cell on each side (zero initialised)
  t_idx l_size = (m_nCellsx + 2) * (m_nCellsy + 2);
  m_h = new t_real[l_size]{};
  m_hu = new t_real[l_size]{};
  m_hv = new t_real[l_size]{};
  m_hTemp = new t_real[l_size]{};
  m_huvTemp = new t_real[l_size]{};
  m_b = new t_real[l_size]{};
}

tsunami_lab::patches::WavePropagation2d::~WavePropagation2d()
{
  delete[] m_h;
  delete[] m_hu;
  delete[] m_hv;
  delete[] m_hTemp;
  delete[] m_huvTemp;
  delete[] m_b;
}

tsunami_lab::t_idx tsunami_lab::patches::WavePropagation2d::getCoord(t_idx i_x, t_idx i_y)
{
  return i_x + i_y * (m_nCellsx + 2);
}

void tsunami_lab::patches::WavePropagation2d::timeStep(t_real i_scaling)
{
  setGhostCellsX();
  // init new cell quantities
  std::copy(m_h, m_h + (m_nCellsx + 2) * (m_nCellsy + 2), m_hTemp);
  std::copy(m_hu, m_hu + (m_nCellsx + 2) * (m_nCellsy + 2), m_huvTemp);

// iterate over edges and update with Riemann solutions in x direction
#pragma omp parallel for
  for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
    for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
    {
      // determine left and right cell-id
      t_idx l_ceL = getCoord(l_ex, l_ey);
      t_idx l_ceR = getCoord(l_ex + 1, l_ey);

      // compute net-updates
      t_real l_netUpdates[2][2];

      if (m_useFWave)
      {
        solvers::FWave::netUpdates(m_hTemp[l_ceL],
                                   m_hTemp[l_ceR],
                                   m_huvTemp[l_ceL],
                                   m_huvTemp[l_ceR],
                                   m_b[l_ceL],
                                   m_b[l_ceR],
                                   l_netUpdates[0],
                                   l_netUpdates[1]);
      }
      else
      {
        solvers::Roe::netUpdates(m_hTemp[l_ceL],
                                 m_hTemp[l_ceR],
                                 m_huvTemp[l_ceL],
                                 m_huvTemp[l_ceR],
                                 l_netUpdates[0],
                                 l_netUpdates[1]);
      }

      // update the cells' quantities
      m_h[l_ceL] -= i_scaling * l_netUpdates[0][0];
      m_hu[l_ceL] -= i_scaling * l_netUpdates[0][1];

      m_h[l_ceR] -= i_scaling * l_netUpdates[1][0];
      m_hu[l_ceR] -= i_scaling * l_netUpdates[1][1];
    }

  setGhostCellsY();

  // init new cell quantities
  std::copy(m_h, m_h + (m_nCellsx + 2) * (m_nCellsy + 2), m_hTemp);
  std::copy(m_hv, m_hv + (m_nCellsx + 2) * (m_nCellsy + 2), m_huvTemp);

// iterate over edges and update with Riemann solutions in y direction
#pragma omp parallel for
  for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
    for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
    {
      // determine top and bottom cell-id
      t_idx l_ceB = getCoord(l_ex, l_ey);
      t_idx l_ceT = getCoord(l_ex, l_ey + 1);

      // compute net-updates
      t_real l_netUpdates[2][2];

      if (m_useFWave)
      {
        solvers::FWave::netUpdates(m_hTemp[l_ceB],
                                   m_hTemp[l_ceT],
                                   m_huvTemp[l_ceB],
                                   m_huvTemp[l_ceT],
                                   m_b[l_ceB],
                                   m_b[l_ceT],
                                   l_netUpdates[0],
                                   l_netUpdates[1]);
      }
      else
      {
        solvers::Roe::netUpdates(m_hTemp[l_ceB],
                                 m_hTemp[l_ceT],
                                 m_huvTemp[l_ceB],
                                 m_huvTemp[l_ceT],
                                 l_netUpdates[0],
                                 l_netUpdates[1]);
      }

      // update the cells' quantities
      m_h[l_ceB] -= i_scaling * l_netUpdates[0][0];
      m_hv[l_ceB] -= i_scaling * l_netUpdates[0][1];

      m_h[l_ceT] -= i_scaling * l_netUpdates[1][0];
      m_hv[l_ceT] -= i_scaling * l_netUpdates[1][1];
    }
}

void tsunami_lab::patches::WavePropagation2d::setGhostCellsX()
{
  t_real *l_h = m_h;
  t_real *l_hu = m_hu;

  // set left boundary
  if (m_boundaryLeft == t_boundary::OPEN)
  {
#pragma GCC ivdep
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_h[getCoord(0, l_y)] = l_h[getCoord(1, l_y)];
      l_hu[getCoord(0, l_y)] = l_hu[getCoord(1, l_y)];
    }
  }

  // set right boundary
  if (m_boundaryRight == t_boundary::OPEN)
  {
#pragma GCC ivdep
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_h[getCoord(m_nCellsx + 1, l_y)] = l_h[getCoord(m_nCellsx, l_y)];
      l_hu[getCoord(m_nCellsx + 1, l_y)] = l_hu[getCoord(m_nCellsx, l_y)];
    }
  }
}

void tsunami_lab::patches::WavePropagation2d::setGhostCellsY()
{
  t_real *l_h = m_h;
  t_real *l_hv = m_hv;

  // set bottom boundary
  if (m_boundaryBottom == t_boundary::OPEN)
  {
#pragma GCC ivdep
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_h[getCoord(l_x, 0)] = l_h[getCoord(l_x, 1)];
      l_hv[getCoord(l_x, 0)] = l_hv[getCoord(l_x, 1)];
    }
  }

  // set top boundary
  if (m_boundaryTop == t_boundary::OPEN)
  {
#pragma GCC ivdep
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_h[getCoord(l_x, m_nCellsy + 1)] = l_h[getCoord(l_x, m_nCellsy)];
      l_hv[getCoord(l_x, m_nCellsy + 1)] = l_hv[getCoord(l_x, m_nCellsy)];
    }
  }
}

void tsunami_lab::patches::WavePropagation2d::initGhostCells()
{
  t_real *l_b = m_b;
  t_real *l_h = m_h;
  t_real *l_hu = m_hu;
  t_real *l_hv = m_hv;

  // set left boundary
  switch (m_boundaryLeft)
  {
  case t_boundary::OPEN:
  {
#pragma GCC ivdep
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_b[getCoord(0, l_y)] = l_b[getCoord(1, l_y)];
    }
    break;
  }
  case t_boundary::WALL:
  {
#pragma GCC ivdep
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_b[getCoord(0, l_y)] = 20;
      l_h[getCoord(0, l_y)] = 0;
      l_hu[getCoord(0, l_y)] = 0;
      l_hv[getCoord(0, l_y)] = 0;
    }
    break;
  }
  }

  // set right boundary
  switch (m_boundaryRight)
  {
  case t_boundary::OPEN:
  {
#pragma GCC ivdep
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_b[getCoord(m_nCellsx + 1, l_y)] = l_b[getCoord(m_nCellsx, l_y)];
    }
    break;
  }
  case t_boundary::WALL:
  {
#pragma GCC ivdep
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_b[getCoord(m_nCellsx + 1, l_y)] = 20;
      l_h[getCoord(m_nCellsx + 1, l_y)] = 0;
      l_hu[getCoord(m_nCellsx + 1, l_y)] = 0;
      l_hv[getCoord(m_nCellsx + 1, l_y)] = 0;
    }
    break;
  }
  }

  // set bottom boundary
  switch (m_boundaryBottom)
  {
  case t_boundary::OPEN:
  {
#pragma GCC ivdep
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_b[getCoord(l_x, 0)] = l_b[getCoord(l_x, 1)];
    }
    break;
  }
  case t_boundary::WALL:
  {
#pragma GCC ivdep
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_b[getCoord(l_x, 0)] = 20;
      l_h[getCoord(l_x, 0)] = 0;
      l_hu[getCoord(l_x, 0)] = 0;
      l_hv[getCoord(l_x, 0)] = 0;
    }
    break;
  }
  }

  // set top boundary
  switch (m_boundaryTop)
  {
  case t_boundary::OPEN:
  {
#pragma GCC ivdep
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_b[getCoord(l_x, m_nCellsy + 1)] = l_b[getCoord(l_x, m_nCellsy)];
    }
    break;
  }
  case t_boundary::WALL:
  {
#pragma GCC ivdep
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_b[getCoord(l_x, m_nCellsy + 1)] = 20;
      l_h[getCoord(l_x, m_nCellsy + 1)] = 0;
      l_hu[getCoord(l_x, m_nCellsy + 1)] = 0;
      l_hv[getCoord(l_x, m_nCellsy + 1)] = 0;
    }
    break;
  }
  }
}