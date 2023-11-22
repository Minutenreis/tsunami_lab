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

TEST_CASE("Test the artificial tsunami setup.", "[ArtificialTsunami2d]")
{
  tsunami_lab::setups::ArtificialTsunami2d l_artificialTsunami;

  // x not in [-500, 500] or y not in [-500, 500]
  REQUIRE(l_artificialTsunami.getHeight(1000, 0) == 100);
  REQUIRE(l_artificialTsunami.getMomentumX(1000, 0) == 0);
  REQUIRE(l_artificialTsunami.getMomentumY(1000, 0) == 0);
  REQUIRE(l_artificialTsunami.getBathymetry(1000, 0) == -100);

  REQUIRE(l_artificialTsunami.getHeight(0, 1000) == 100);
  REQUIRE(l_artificialTsunami.getMomentumX(0, 1000) == 0);
  REQUIRE(l_artificialTsunami.getMomentumY(0, 1000) == 0);
  REQUIRE(l_artificialTsunami.getBathymetry(0, 1000) == -100);

  // x = y = 0
  // f(x) = sin(pi)  = 0
  // d(x,y) = 5 * f(x)g(y) = 0
  REQUIRE(l_artificialTsunami.getHeight(0, 0) == 100);
  REQUIRE(l_artificialTsunami.getMomentumX(0, 0) == 0);
  REQUIRE(l_artificialTsunami.getMomentumY(0, 0) == 0);
  REQUIRE(l_artificialTsunami.getBathymetry(0, 0) == -100);

  // x = 250, y = 250
  // f(x) = sin(1.5 pi)  = 1
  // g(x) = -0.25 + 1 = 0.75
  // d(x,y) = 5 * f(x)g(y) = 3.75
  REQUIRE(l_artificialTsunami.getHeight(250, 250) == 100);
  REQUIRE(l_artificialTsunami.getMomentumX(250, 250) == 0);
  REQUIRE(l_artificialTsunami.getMomentumY(250, 250) == 0);
  REQUIRE(l_artificialTsunami.getBathymetry(250, 250) == -96.25);
}