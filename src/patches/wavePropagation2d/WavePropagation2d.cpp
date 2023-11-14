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

tsunami_lab::patches::WavePropagation2d::WavePropagation2d(t_idx i_nCellsx,
                                                           t_idx i_nCellsy,
                                                           bool i_useFWave,
                                                           t_boundary i_boundaryLeft,
                                                           t_boundary i_boundaryRight,
                                                           t_boundary i_boundaryBottom,
                                                           t_boundary i_boundaryTop)
{
  m_nCellsx = i_nCellsx;
  m_nCellsy = i_nCellsy;
  m_useFWave = i_useFWave;
  m_boundaryLeft = i_boundaryLeft;
  m_boundaryRight = i_boundaryRight;
  m_boundaryBottom = i_boundaryBottom;
  m_boundaryTop = i_boundaryTop;

  // allocate memory including a single ghost cell on each side
  for (unsigned short l_st = 0; l_st < 2; l_st++)
  {
    m_h[l_st] = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
    m_hu[l_st] = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
    m_hv[l_st] = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
  }
  m_b = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];

  // init to zero
  for (unsigned short l_st = 0; l_st < 2; l_st++)
  {
    for (t_idx l_ce = 0; l_ce < (m_nCellsx + 2) * (m_nCellsy + 2); l_ce++)
    {
      m_h[l_st][l_ce] = 0;
      m_hu[l_st][l_ce] = 0;
      m_hv[l_st][l_ce] = 0;
      m_b[l_ce] = 0;
    }
  }
}

tsunami_lab::patches::WavePropagation2d::~WavePropagation2d()
{
  for (unsigned short l_st = 0; l_st < 2; l_st++)
  {
    delete[] m_h[l_st];
    delete[] m_hu[l_st];
    delete[] m_hv[l_st];
  }
  delete[] m_b;
}

tsunami_lab::t_idx tsunami_lab::patches::WavePropagation2d::getCoord(t_idx i_x, t_idx i_y)
{
  return i_x + i_y * (m_nCellsx + 2);
}

void tsunami_lab::patches::WavePropagation2d::timeStep(t_real i_scaling)
{
  setGhostOutflow();
  // pointers to old and new data
  t_real *l_hOld = m_h[m_step];
  t_real *l_huOld = m_hu[m_step];
  t_real *l_hvOld = m_hv[m_step];

  m_step = (m_step + 1) % 2;
  t_real *l_hNew = m_h[m_step];
  t_real *l_huNew = m_hu[m_step];
  t_real *l_hvNew = m_hv[m_step];

  // init new cell quantities
  for (t_idx l_cx = 1; l_cx < m_nCellsx + 1; l_cx++)
    for (t_idx l_cy = 1; l_cy < m_nCellsy + 1; l_cy++)
    {
      l_hNew[getCoord(l_cx, l_cy)] = l_hOld[getCoord(l_cx, l_cy)];
      l_huNew[getCoord(l_cx, l_cy)] = l_huOld[getCoord(l_cx, l_cy)];
      l_hvNew[getCoord(l_cx, l_cy)] = l_hvOld[getCoord(l_cx, l_cy)];
    }

  // iterate over edges and update with Riemann solutions in x direction
  for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
    for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
    {
      // determine left and right cell-id
      t_idx l_ceL = getCoord(l_ex, l_ey);
      t_idx l_ceR = getCoord(l_ex + 1, l_ey);

      // compute net-updates
      t_real l_netUpdates[2][2];

      if (m_useFWave)
      {
        solvers::FWave::netUpdates(l_hOld[l_ceL],
                                   l_hOld[l_ceR],
                                   l_huOld[l_ceL],
                                   l_huOld[l_ceR],
                                   m_b[l_ceL],
                                   m_b[l_ceR],
                                   l_netUpdates[0],
                                   l_netUpdates[1]);
      }
      else
      {
        solvers::Roe::netUpdates(l_hOld[l_ceL],
                                 l_hOld[l_ceR],
                                 l_huOld[l_ceL],
                                 l_huOld[l_ceR],
                                 l_netUpdates[0],
                                 l_netUpdates[1]);
      }

      // update the cells' quantities
      l_hNew[l_ceL] -= i_scaling * l_netUpdates[0][0];
      l_huNew[l_ceL] -= i_scaling * l_netUpdates[0][1];

      l_hNew[l_ceR] -= i_scaling * l_netUpdates[1][0];
      l_huNew[l_ceR] -= i_scaling * l_netUpdates[1][1];
    }

  setGhostOutflow();
  // pointers to old and new data
  l_hOld = m_h[m_step];
  l_huOld = m_hu[m_step];
  l_hvOld = m_hv[m_step];

  m_step = (m_step + 1) % 2;
  l_hNew = m_h[m_step];
  l_huNew = m_hu[m_step];
  l_hvNew = m_hv[m_step];

  // init new cell quantities
  for (t_idx l_cx = 1; l_cx < m_nCellsx + 1; l_cx++)
    for (t_idx l_cy = 1; l_cy < m_nCellsy + 1; l_cy++)
    {
      l_hNew[getCoord(l_cx, l_cy)] = l_hOld[getCoord(l_cx, l_cy)];
      l_huNew[getCoord(l_cx, l_cy)] = l_huOld[getCoord(l_cx, l_cy)];
      l_hvNew[getCoord(l_cx, l_cy)] = l_hvOld[getCoord(l_cx, l_cy)];
    }

  // iterate over edges and update with Riemann solutions in y direction
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
        solvers::FWave::netUpdates(l_hOld[l_ceB],
                                   l_hOld[l_ceT],
                                   l_hvOld[l_ceB],
                                   l_hvOld[l_ceT],
                                   m_b[l_ceB],
                                   m_b[l_ceT],
                                   l_netUpdates[0],
                                   l_netUpdates[1]);
      }
      else
      {
        solvers::Roe::netUpdates(l_hOld[l_ceB],
                                 l_hOld[l_ceT],
                                 l_hvOld[l_ceB],
                                 l_hvOld[l_ceT],
                                 l_netUpdates[0],
                                 l_netUpdates[1]);
      }

      // update the cells' quantities
      l_hNew[l_ceB] -= i_scaling * l_netUpdates[0][0];
      l_hvNew[l_ceB] -= i_scaling * l_netUpdates[0][1];

      l_hNew[l_ceT] -= i_scaling * l_netUpdates[1][0];
      l_hvNew[l_ceT] -= i_scaling * l_netUpdates[1][1];
    }
}

void tsunami_lab::patches::WavePropagation2d::setGhostOutflow()
{
  t_real *l_h = m_h[m_step];
  t_real *l_hu = m_hu[m_step];
  t_real *l_hv = m_hv[m_step];
  t_real *l_b = m_b;

  // set left boundary
  switch (m_boundaryLeft)
  {
  case t_boundary::OPEN:
  {
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_h[getCoord(0, l_y)] = l_h[getCoord(1, l_y)];
      l_hu[getCoord(0, l_y)] = l_hu[getCoord(1, l_y)];
      l_hv[getCoord(0, l_y)] = l_hv[getCoord(1, l_y)];
      l_b[getCoord(0, l_y)] = l_b[getCoord(1, l_y)];
    }
    break;
  }
  case t_boundary::WALL:
  {
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_h[getCoord(0, l_y)] = 0;
      l_hu[getCoord(0, l_y)] = 0;
      l_hv[getCoord(0, l_y)] = 0;
      l_b[getCoord(0, l_y)] = 20;
    }
    break;
  }
  }

  // set right boundary
  switch (m_boundaryRight)
  {
  case t_boundary::OPEN:
  {
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_h[getCoord(m_nCellsx + 1, l_y)] = l_h[getCoord(m_nCellsx, l_y)];
      l_hu[getCoord(m_nCellsx + 1, l_y)] = l_hu[getCoord(m_nCellsx, l_y)];
      l_hv[getCoord(m_nCellsx + 1, l_y)] = l_hv[getCoord(m_nCellsx, l_y)];
      l_b[getCoord(m_nCellsx + 1, l_y)] = l_b[getCoord(m_nCellsx, l_y)];
    }
    break;
  }
  case t_boundary::WALL:
  {
    for (t_idx l_y = 1; l_y < m_nCellsy + 1; l_y++)
    {
      l_h[getCoord(m_nCellsx + 1, l_y)] = 0;
      l_hu[getCoord(m_nCellsx + 1, l_y)] = 0;
      l_hv[getCoord(m_nCellsx + 1, l_y)] = 0;
      l_b[getCoord(m_nCellsx + 1, l_y)] = 20;
    }
    break;
  }
  }

  // set top boundary
  switch (m_boundaryTop)
  {
  case t_boundary::OPEN:
  {
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_h[getCoord(l_x, 0)] = l_h[getCoord(l_x, 1)];
      l_hu[getCoord(l_x, 0)] = l_hu[getCoord(l_x, 1)];
      l_hv[getCoord(l_x, 0)] = l_hv[getCoord(l_x, 1)];
      l_b[getCoord(l_x, 0)] = l_b[getCoord(l_x, 1)];
    }
    break;
  }
  case t_boundary::WALL:
  {
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_h[getCoord(l_x, 0)] = 0;
      l_hu[getCoord(l_x, 0)] = 0;
      l_hv[getCoord(l_x, 0)] = 0;
      l_b[getCoord(l_x, 0)] = 20;
    }
    break;
  }
  }

  // set top boundary
  switch (m_boundaryTop)
  {
  case t_boundary::OPEN:
  {
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_h[getCoord(l_x, m_nCellsx + 1)] = l_h[getCoord(l_x, m_nCellsx)];
      l_hu[getCoord(l_x, m_nCellsx + 1)] = l_hu[getCoord(l_x, m_nCellsx)];
      l_hv[getCoord(l_x, m_nCellsx + 1)] = l_hv[getCoord(l_x, m_nCellsx)];
      l_b[getCoord(l_x, m_nCellsx + 1)] = l_b[getCoord(l_x, m_nCellsx)];
    }
    break;
  }
  case t_boundary::WALL:
  {
    for (t_idx l_x = 1; l_x < m_nCellsx + 1; l_x++)
    {
      l_h[getCoord(l_x, m_nCellsx + 1)] = 0;
      l_hu[getCoord(l_x, m_nCellsx + 1)] = 0;
      l_hv[getCoord(l_x, m_nCellsx + 1)] = 0;
      l_b[getCoord(l_x, m_nCellsx + 1)] = 20;
    }
    break;
  }
  }
}