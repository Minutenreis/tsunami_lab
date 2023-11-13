/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional dam break problem.
 **/
#ifndef TSUNAMI_LAB_SETUPS_DAM_BREAK_2D_H
#define TSUNAMI_LAB_SETUPS_DAM_BREAK_2D_H

#include <cmath>
#include "../Setup.h"

namespace tsunami_lab
{
  namespace setups
  {
    class DamBreak2d;
  }
}

/**
 * 2d dam break setup.
 **/
class tsunami_lab::setups::DamBreak2d : public Setup
{
private:
  //! height on the left side
  t_real m_heightLeft = 0;

  //! height on the right side
  t_real m_heightRight = 0;

  //! location of the dam
  t_real m_locationDam = 0;

public:
  /**
   * Constructor.
   *
   * @param i_heightLeft water height on the left side of the dam.
   * @param i_heightRight water height on the right side of the dam.
   * @param i_locationDam location (x-coordinate) of the dam.
   **/
  DamBreak2d(t_real i_heightLeft,
             t_real i_heightRight,
             t_real i_locationDam);

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
   * @return bathymetry
   */
  t_real getBathymetry(t_real,
                       t_real) const;
};

#endif