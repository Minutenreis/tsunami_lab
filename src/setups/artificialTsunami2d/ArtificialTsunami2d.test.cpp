/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional artificial tsunami problem.
 **/
#include <catch2/catch.hpp>
#include "ArtificialTsunami2d.h"

TEST_CASE("Test the two-dimensional dam break setup.", "[ArtificialTsunami2d]")
{
  tsunami_lab::setups::ArtificialTsunami2d l_artificialTsunami;

  // sqrt(i_x*i_x + i_y*i_y) < 10
  REQUIRE(l_artificialTsunami.getHeight(0, 0) == 10);

  REQUIRE(l_artificialTsunami.getMomentumX(0, 0) == 0);

  REQUIRE(l_artificialTsunami.getMomentumY(0, 0) == 0);

  REQUIRE(l_artificialTsunami.getBathymetry(0, 0) == -10);

  // sqrt(i_x*i_x + i_y*i_y) > 10
  REQUIRE(l_artificialTsunami.getHeight(10, 10) == 5);

  REQUIRE(l_artificialTsunami.getMomentumX(10, 10) == 0);

  REQUIRE(l_artificialTsunami.getMomentumY(10, 10) == 0);

  REQUIRE(l_artificialTsunami.getBathymetry(10, 10) == -10);

  REQUIRE(l_artificialTsunami.getBathymetry(31, 0) == 10);
}