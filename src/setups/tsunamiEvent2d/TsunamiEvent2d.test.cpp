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
  tsunami_lab::setups::TsunamiEvent2d* l_tsunamievent2d = new tsunami_lab::setups::TsunamiEvent2d(
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

  //check if the getDisplacement function works
}