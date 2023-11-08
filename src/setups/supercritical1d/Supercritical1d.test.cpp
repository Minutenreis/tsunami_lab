/**Supercritical1d
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Tests the supercritical setup.
 **/
#include <catch2/catch.hpp>
#include "Supercritical1d.h"

TEST_CASE("Test the one-dimensional supercritical setup.", "[Supercritical1d]")
{
  tsunami_lab::setups::Supercritical1d l_supercritical(25,
                                                 55,
                                                 3);
  // x = 2
  REQUIRE(l_supercritical.getHeight(2, 0) == 0.33);

  REQUIRE(l_supercritical.getMomentumX(2, 0) == 0.18);

  REQUIRE(l_supercritical.getMomentumY(2, 0) == 0);

  REQUIRE(l_supercritical.getBathymetry(2, 0) == -0.33);


  // x element of 8-12
  REQUIRE(l_supercritical.getHeight(10, 0) == 0.13);

  REQUIRE(l_supercritical.getMomentumX(10, 0) == 0.18);

  REQUIRE(l_supercritical.getMomentumY(4, 0) == 0);

  REQUIRE(l_supercritical.getBathymetry(10, 0) == -0.13);

  // x element over 25

  REQUIRE(l_supercritical.getHeight(30, 0) == 25);

  REQUIRE(l_supercritical.getMomentumX(30, 0) == 55);

  REQUIRE(l_supercritical.getMomentumY(30, 0) == 0);

  REQUIRE(l_supercritical.getBathymetry(30, 0) == -0.33);
}