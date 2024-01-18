/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional wave propagation patch.
 **/
#ifndef TSUNAMI_LAB_PATCHES_WAVE_PROPAGATION_2D
#define TSUNAMI_LAB_PATCHES_WAVE_PROPAGATION_2D

#include "../WavePropagation.h"

namespace tsunami_lab
{
  namespace patches
  {
    class WavePropagation2d;
  }
}

class tsunami_lab::patches::WavePropagation2d : public WavePropagation
{
private:
  //! current step which indicates the active values in the arrays below
  // unsigned short m_step = 0;

  //! number of cells discretizing the computational domain in x direction
  const t_idx m_nCellsx = 0;

  //! number of cells discretizing the computational domain in y direction
  const t_idx m_nCellsy = 0;

  //! bool if FWave solver is used
  const bool m_useFWave = true;

  //! left boundary
  const t_boundary m_boundaryLeft = t_boundary::OPEN;

  //! right boundary
  const t_boundary m_boundaryRight = t_boundary::OPEN;

  //! bottom boundary
  const t_boundary m_boundaryBottom = t_boundary::OPEN;

  //! top boundary
  const t_boundary m_boundaryTop = t_boundary::OPEN;

  //! water heights for the current and next time step for all cells
  //! access array like m_h[i_x  + i_y  * (m_nCells + 2)]
  t_real *m_h = nullptr;

  //! momenta for the current and next time step for all cells in x-direction
  //! access array like m_hu[i_x + i_y  * (m_nCells + 2)]
  t_real *m_hu = nullptr;

  //! momenta for the current and next time step for all cells in y-direction
  //! access array like m_hv[i_x  + i_y  * (m_nCells + 2)]
  t_real *m_hv = nullptr;

  //! temp array for height
  t_real *m_hTemp = nullptr;

  //! temp array for momentum
  t_real *m_huvTemp = nullptr;

  //! bathymetry for the current and next time step for all cells
  //! access array like m_h[i_x + i_y  * (m_nCells + 2)]
  t_real *m_b = nullptr;

  /**
   * @brief get Coordinate from x and y index
   *
   * @param i_x x index
   * @param i_y y index
   * @return coordinate
   */
  t_idx getCoord(t_idx i_x, t_idx i_y);

  /**
   * Sets the values of the ghost cells according to t_boundary set in the class for the X Sweep.
   **/
  void setGhostCellsX();

  /**
   * Sets the values of the ghost cells according to t_boundary set in the class for the Y Sweep.
   **/
  void setGhostCellsY();

public:
  /**
   * Constructs the 1d wave propagation solver.
   *
   * @param i_nCellsx number of cells in x direction.
   * @param i_nCellsy number of cells in y direction.
   * @param i_useFWave bool: true if FWave solver should be used, false if Roe solver should be used.
   * @param i_boundaryLeft left boundary condition.
   * @param i_boundaryRight right boundary condition.
   * @param i_boundaryBottom bottom boundary condition.
   * @param i_boundaryTop top boundary condition.
   **/
  WavePropagation2d(t_idx i_nCellsx,
                    t_idx i_nCellsy,
                    bool i_useFWave,
                    t_boundary i_boundaryLeft,
                    t_boundary i_boundaryRight,
                    t_boundary i_boundaryBottom,
                    t_boundary i_boundaryTop);

  /**
   * Destructor which frees all allocated memory.
   **/
  ~WavePropagation2d();

  /**
   * Performs a time step.
   *
   * @param i_scaling scaling of the time step (dt / dx).
   **/
  void timeStep(t_real i_scaling);

  /**
   * Gets the stride in y-direction. x-direction is stride-1.
   *
   * @return stride in y-direction.
   **/
  t_idx getStride()
  {
    return m_nCellsx + 2;
  }

  /**
   * @brief Gets number of ghost cells in x-direction.
   *
   * @return number of ghost cells in x-direction.
   */
  t_idx getGhostCellsX()
  {
    return 1;
  }

  /**
   * @brief Gets number of ghost cells in y-direction.
   *
   * @return number of ghost cells in y-direction.
   */
  t_idx getGhostCellsY()
  {
    return 1;
  }

  /**
   * Gets cells' water heights.
   *
   * @return water heights.
   */
  t_real const *getHeight()
  {
    return m_h;
  }

  /**
   * Gets the cells' momenta in x-direction.
   *
   * @return momenta in x-direction.
   **/
  t_real const *getMomentumX()
  {
    return m_hu;
  }

  /**
   * Dummy function which returns a nullptr.
   **/
  t_real const *getMomentumY()
  {
    return m_hv;
  }

  /**
   * @brief Gets the cells bathymetry.
   *
   * @return bathymetry.
   */
  t_real const *getBathymetry()
  {
    return m_b;
  }

  /**
   * Sets the height of the cell to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_h water height.
   **/
  void setHeight(t_idx i_ix,
                 t_idx i_iy,
                 t_real i_h)
  {
    m_h[getCoord(i_ix + 1, i_iy + 1)] = i_h;
  }

  /**
   * Sets the momentum in x-direction to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_hu momentum in x-direction.
   **/
  void setMomentumX(t_idx i_ix,
                    t_idx i_iy,
                    t_real i_hu)
  {
    m_hu[getCoord(i_ix + 1, i_iy + 1)] = i_hu;
  }

  /**
   * Dummy function since there is no y-momentum in the 1d solver.
   **/
  void setMomentumY(t_idx i_ix,
                    t_idx i_iy,
                    t_real i_hv)
  {
    m_hv[getCoord(i_ix + 1, i_iy + 1)] = i_hv;
  };

  /**
   * @brief Sets the bathymetry of the cell to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_b bathymetry.
   */
  void setBathymetry(t_idx i_ix,
                     t_idx i_iy,
                     t_real i_b)
  {
    m_b[getCoord(i_ix + 1, i_iy + 1)] = i_b;
  }

  /**
   * @brief Initializes Bathymetry Ghost Cells
   */
  void initGhostCells();
};

#endif