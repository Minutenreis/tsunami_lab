/**
 * @author Alexander Breuer (alex.breuer AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Base class of the wave propagation patches.
 **/
#ifndef TSUNAMI_LAB_PATCHES_WAVE_PROPAGATION
#define TSUNAMI_LAB_PATCHES_WAVE_PROPAGATION

#include "../constants.h"

namespace tsunami_lab
{
  namespace patches
  {
    class WavePropagation;
  }
}

class tsunami_lab::patches::WavePropagation
{
public:
  /**
   * Virtual destructor for base class.
   **/
  virtual ~WavePropagation(){};

  /**
   * Performs a time step.
   *
   * @param i_scaling scaling of the time step.
   **/
  virtual void timeStep(t_real i_scaling) = 0;

  /**
   * Gets the stride in y-direction. x-direction is stride-1.
   *
   * @return stride in y-direction.
   **/
  virtual t_idx getStride() = 0;

  /**
   * @brief Gets number of ghost cells in x-direction.
   *
   * @return number of ghost cells in x-direction.
   */
  virtual t_idx getGhostCellsX() = 0;

  /**
   * @brief Gets number of ghost cells in y-direction.
   *
   * @return number of ghost cells in y-direction.
   */
  virtual t_idx getGhostCellsY() = 0;

  /**
   * Gets cells' water heights.
   *
   * @return water heights.
   */
  virtual t_real const *getHeight() = 0;

  /**
   * Gets the cells' momenta in x-direction.
   *
   * @return momenta in x-direction.
   **/
  virtual t_real const *getMomentumX() = 0;

  /**
   * Gets the cells' momenta in y-direction.
   *
   * @return momenta in y-direction.
   **/
  virtual t_real const *getMomentumY() = 0;

  /**
   * @brief Gets the cells bathymetry.
   *
   * @return bathymetry.
   */
  virtual t_real const *getBathymetry() = 0;

  /**
   * Sets the height of the cell to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_iy id of the cell in y-direction.
   * @param i_h water height.
   **/
  virtual void setHeight(t_idx i_ix,
                         t_idx i_iy,
                         t_real i_h) = 0;

  /**
   * Sets the momentum in x-direction to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_iy id of the cell in y-direction.
   * @param i_hu momentum in x-direction.
   **/
  virtual void setMomentumX(t_idx i_ix,
                            t_idx i_iy,
                            t_real i_hu) = 0;

  /**
   * Sets the momentum in y-direction to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_iy id of the cell in y-direction.
   * @param i_hv momentum in y-direction.
   **/
  virtual void setMomentumY(t_idx i_ix,
                            t_idx i_iy,
                            t_real i_hv) = 0;

  /**
   * Sets the bathymetry to the given value.
   *
   * @param i_ix id of the cell in x-direction.
   * @param i_iy id of the cell in y-direction.
   * @param i_b bathymetry.
   **/
  virtual void setBathymetry(t_idx i_ix,
                             t_idx i_iy,
                             t_real i_b) = 0;

  /**
   * @brief Initializes the ghost cells. You can't use set Operations afterwards.
   *
   */
  virtual void initGhostCells() = 0;

  /**
   * @brief Prepares Data Access (you can use get Operations afterwards until the next time step)
   */
  virtual void prepareDataAccess() = 0;
};

#endif