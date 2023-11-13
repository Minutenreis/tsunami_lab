/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional dam break problem.
 **/
#include <catch2/catch.hpp>
#include "DamBreak2d.h"

TEST_CASE( "Test the one-dimensional dam break setup.", "[DamBreak2d]" ) {
  tsunami_lab::setups::DamBreak2d l_damBreak( 25,
                                              55,
                                               3 );

  // sqrt(i_x*i_x + i_y*i_y) < 10
  REQUIRE( l_damBreak.getHeight( 0, 0 ) == 10 );

  REQUIRE( l_damBreak.getMomentumX( 0, 0 ) == 0 );

  REQUIRE( l_damBreak.getMomentumY( 0, 0 ) == 0 );

  REQUIRE( l_damBreak.getBathymetry( 0, 0 ) == 0 );

  // sqrt(i_x*i_x + i_y*i_y) > 10
  REQUIRE( l_damBreak.getHeight( 10, 10 ) == 5 );

  REQUIRE( l_damBreak.getMomentumX( 10, 10 ) == 0 );

  REQUIRE( l_damBreak.getMomentumY( 10, 10 ) == 0 );

  REQUIRE( l_damBreak.getBathymetry( 10, 10 ) == 0 );

}