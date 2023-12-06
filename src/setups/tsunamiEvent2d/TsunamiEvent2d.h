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

  //! length of bathymetry in x direction.
  t_idx m_nbX;

  //! length of bathymetry in y direction.
  t_idx m_nbY;

  //! x-value array of the bathymetry.
  t_real *m_bathymetryX;

  //! y-value array of the bathymetry.
  t_real *m_bathymetryY;

  //! bathymetry array.
  t_real *m_bathymetry;

  //! cell size in x direction for bathymetry
  t_real m_stepBathX;

  //! cell size in y direction for bathymetry
  t_real m_stepBathY;

  //! offset in x direction for bathymetry
  t_real m_offsetBathX;

  //! offset in y direction for bathymetry
  t_real m_offsetBathY;

  //! length of displacement in x direction.
  t_idx m_ndX;

  //! length of displacement in y direction.
  t_idx m_ndY;

  //! x-value array of the displacement.
  t_real *m_displacementX;

  //! y-value array of the displacement.
  t_real *m_displacementY;

  //! displacement array.
  t_real *m_displacement;

  //! cell size in x direction for displacement
  t_real m_stepDisplX;

  //! cell size in y direction for displacement
  t_real m_stepDisplY;

  //! offset in x direction for displacement
  t_real m_offsetDisplX;

  //! offset in y direction for displacement
  t_real m_offsetDisplY;

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
   * @brief Constructor
   *
   * @param i_displacement path to the displacement file
   * @param i_bathymetry path to the bathymetry file
   * @param o_width width of the domain
   * @param o_height height of the domain
   * @param o_offsetX offset in x-direction
   * @param o_offsetY offset in y-direction
   */
  TsunamiEvent2d(char *i_displacement,
                 char *i_bathymetry,
                 t_real *o_width,
                 t_real *o_height,
                 t_real *o_offsetX,
                 t_real *o_offsetY);

  /**
   * @brief Destructor
   *
   */
  ~TsunamiEvent2d();

  /**
   * Gets the water height at a given point.
   *
   * @param i_x x-coordinate of the queried point.
   * @param i_y y-coordinate of the queried point.
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