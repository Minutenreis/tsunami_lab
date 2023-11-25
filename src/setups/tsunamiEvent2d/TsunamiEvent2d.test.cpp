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
#include "string"

TEST_CASE("Test the two-dimensional tsunamiEvent setup.", "[TsunamiEvent2d]")
{
  tsunami_lab::t_real o_width;
  tsunami_lab::t_real o_height;
  tsunami_lab::t_real o_offsetX;
  tsunami_lab::t_real o_offsetY;

  std::string l_displacement = "src/data/testDispl.nc";
  std::string l_bathymetry = "src/data/testBathymetry.nc";

  //create tsunami2d constructor
  tsunami_lab::setups::TsunamiEvent2d* l_tsunamiEvent2d = new tsunami_lab::setups::TsunamiEvent2d(
    l_displacement.data(),
      l_bathymetry.data(),
                 &o_width,
                &o_height,
               &o_offsetX,
               &o_offsetY);

  //check if the constructor works
  REQUIRE(o_width == 10);
  REQUIRE(o_height == 5);
  REQUIRE(o_offsetX == 0);
  REQUIRE(o_offsetY == 0);

  //test momentumX, should be 0 everywhere
  tsunami_lab::t_real l_momentumX = l_tsunamiEvent2d->getMomentumX(0,0);
  REQUIRE(l_momentumX == 0);

  //test momentumY, should be 0 everywhere
  tsunami_lab::t_real l_momentumY = l_tsunamiEvent2d->getMomentumY(0,0);
  REQUIRE(l_momentumY == 0);

  //check if the getDisplacement function works
  tsunami_lab::t_real l_displacementValue = l_tsunamiEvent2d->getDisplacement(3,1);
  REQUIRE(l_displacementValue == 0);
  tsunami_lab::t_real l_displacementValue = l_tsunamiEvent2d->getDisplacement(7,1);
  REQUIRE(l_displacementValue == 4);
  tsunami_lab::t_real l_displacementValue = l_tsunamiEvent2d->getDisplacement(3,2);
  REQUIRE(l_displacementValue == 5);
  tsunami_lab::t_real l_displacementValue = l_tsunamiEvent2d->getDisplacement(7,2);
  REQUIRE(l_displacementValue == 9);
  tsunami_lab::t_real l_displacementValue = l_tsunamiEvent2d->getDisplacement(3,3);
  REQUIRE(l_displacementValue == 10);
  tsunami_lab::t_real l_displacementValue = l_tsunamiEvent2d->getDisplacement(7,3);
  REQUIRE(l_displacementValue == 14);

  //check if the getBathymetry function works
  tsunami_lab::t_real l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(0,0);
  REQUIRE(l_bathymetryValue == -10);
  tsunami_lab::t_real l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(9,0);
  REQUIRE(l_bathymetryValue == -10);
  tsunami_lab::t_real l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(5,2);
  REQUIRE(l_bathymetryValue == -10);
  tsunami_lab::t_real l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(9,4);
  REQUIRE(l_bathymetryValue == -10);
}