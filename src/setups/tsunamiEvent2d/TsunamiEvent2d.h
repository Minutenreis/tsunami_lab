/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional tsunami event problem.
 **/
#ifndef TSUNAMI_LAB_SETUPS_TSUNAMI_EVENT_2D_H
#define TSUNAMI_LAB_SETUPS_TSUNAMI_EVENT_2D_H

#include <cmath>
#include "../Setup.h"

namespace tsunami_lab
{
  namespace setups
  {
    class TsunamiEvent2d;
  }
}

/**
 * Two-dimensional articial tsunami setup.
 **/
class tsunami_lab::setups::TsunamiEvent2d : public Setup
{
private:
  //! delta for heights
  t_real m_delta = 20;

  /**
   * @brief Gets displacement
   *
   * @param i_x x-coordinate of the queried point.
   * @param i_y y-coordinate of the queried point.
   */
  t_real getDisplacement(t_real i_x,
                         t_real i_y) const;

  /**
   * @brief Get the Bathymetry of the NC file
   *
   * @param i_x x-coordinate of the queried point
   * @param i_y y-coordinate of the queried point
   * @return bathymetry
   */
  t_real getBathymetryBin(t_real i_x, t_real i_y) const;

public:
  /**
   * Gets the water height at a given point.
   *
   * @param i_x x-coordinate of the queried point.
   * @return height at the given point.
   **/
  t_real getHeight(t_real i_x,
                   t_real i_y) const;

  /**
   * Gets the momentum in x-direction.
   *
   * @return momentum in x-direction.
   **/
  t_real getMomentumX(t_real,
                      t_real) const;

  /**
   * Gets the momentum in y-direction.
   *
   * @return momentum in y-direction.
   **/
  t_real getMomentumY(t_real,
                      t_real) const;

  /**
   * @brief Gets the bathymetry
   *
   * @param i_x x-coordinate of the queried point.
   * @param i_y y-coordinate of the queried point.
   * @return bathymetry
   */
  t_real getBathymetry(t_real i_x,
                       t_real i_y) const;
};

#endif