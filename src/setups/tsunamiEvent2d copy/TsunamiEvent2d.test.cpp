/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional tsunami event problem.
 **/
#include <catch2/catch.hpp>
#include "TsunamiEvent2d.h"

TEST_CASE("Test the two-dimensional dam break setup.", "[TsunamiEvent2d]")
{
  tsunami_lab::setups::TsunamiEvent2d l_tsunamievent;

  // sqrt(i_x*i_x + i_y*i_y) < 10
  REQUIRE(l_tsunamievent.getHeight(0, 0) == 10);

  REQUIRE(l_tsunamievent.getMomentumX(0, 0) == 0);

  REQUIRE(l_tsunamievent.getMomentumY(0, 0) == 0);

  REQUIRE(l_tsunamievent.getBathymetry(0, 0) == -10);

  // sqrt(i_x*i_x + i_y*i_y) > 10
  REQUIRE(l_tsunamievent.getHeight(10, 10) == 5);

  REQUIRE(l_tsunamievent.getMomentumX(10, 10) == 0);

  REQUIRE(l_tsunamievent.getMomentumY(10, 10) == 0);

  REQUIRE(l_tsunamievent.getBathymetry(10, 10) == -10);

  REQUIRE(l_tsunamievent.getBathymetry(31, 0) == 10);
}