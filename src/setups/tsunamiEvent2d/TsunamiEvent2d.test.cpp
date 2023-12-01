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
#include "../artificialTsunami2d/ArtificialTsunami2d.h"
#include "string"

TEST_CASE("Test the two-dimensional tsunamiEvent setup.", "[TsunamiEvent2d]")
{
  tsunami_lab::t_real o_width;
  tsunami_lab::t_real o_height;
  tsunami_lab::t_real o_offsetX;
  tsunami_lab::t_real o_offsetY;

  std::string l_displacement = "src/data/testDispl.nc";
  std::string l_bathymetry = "src/data/testBathymetry.nc";

  // create tsunami2d constructor
  tsunami_lab::setups::TsunamiEvent2d *l_tsunamiEvent2d = new tsunami_lab::setups::TsunamiEvent2d(
      l_displacement.data(),
      l_bathymetry.data(),
      &o_width,
      &o_height,
      &o_offsetX,
      &o_offsetY);

  // check if the constructor works
  REQUIRE(o_width == Approx(9));
  REQUIRE(o_height == Approx(4));
  REQUIRE(o_offsetX == 0);
  REQUIRE(o_offsetY == 0);

  // test momentumX, should be 0 everywhere
  tsunami_lab::t_real l_momentumX = l_tsunamiEvent2d->getMomentumX(0, 0);
  REQUIRE(l_momentumX == Approx(0));

  // test momentumY, should be 0 everywhere
  tsunami_lab::t_real l_momentumY = l_tsunamiEvent2d->getMomentumY(0, 0);
  REQUIRE(l_momentumY == Approx(0));

  // check if the getBathymetry function works
  tsunami_lab::t_real l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(0, 0);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(9, 0);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(0, 4);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(9, 4);
  REQUIRE(l_bathymetryValue == Approx(-20));

  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(-2, -2);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(2.9, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(3.1, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(4, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-19));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(5, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-18));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(6, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-17));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(6.9, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-16));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(7.1, 1.1);
  REQUIRE(l_bathymetryValue == Approx(-20));

  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(2.9, 2);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(3.1, 2);
  REQUIRE(l_bathymetryValue == Approx(-15));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(4, 2);
  REQUIRE(l_bathymetryValue == Approx(-14));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(5, 2);
  REQUIRE(l_bathymetryValue == Approx(-13));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(6, 2);
  REQUIRE(l_bathymetryValue == Approx(-12));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(6.9, 2);
  REQUIRE(l_bathymetryValue == Approx(-11));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(7.1, 2);
  REQUIRE(l_bathymetryValue == Approx(-20));

  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(2.9, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(3.1, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-10));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(4, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-9));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(5, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-8));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(6, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-7));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(6.9, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-6));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(7.1, 2.9);
  REQUIRE(l_bathymetryValue == Approx(-20));
  l_bathymetryValue = l_tsunamiEvent2d->getBathymetry(30, 30);
  REQUIRE(l_bathymetryValue == Approx(-20));

  // check if getHeight works correct in combination with displacements, should be 20 everywhere
  tsunami_lab::t_real l_heightValue = l_tsunamiEvent2d->getHeight(2, 1);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(3, 1);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(3, 2);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(3, 3);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(5, 1);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(5, 2);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(5, 3);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(7, 1);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(7, 2);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(7, 3);
  REQUIRE(l_heightValue == 20);

  l_heightValue = l_tsunamiEvent2d->getHeight(0, 0);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(0, 4);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(9, 0);
  REQUIRE(l_heightValue == 20);
  l_heightValue = l_tsunamiEvent2d->getHeight(9, 4);
  REQUIRE(l_heightValue == 20);

  delete l_tsunamiEvent2d;
}

TEST_CASE("Test Artificial Tsunami2d Data", "[Tsunami2dEventArtificialData]")
{
  tsunami_lab::t_real o_width;
  tsunami_lab::t_real o_height;
  tsunami_lab::t_real o_offsetX;
  tsunami_lab::t_real o_offsetY;

  std::string l_displacement = "src/data/artificialtsunami_displ_1000.nc";
  std::string l_bathymetry = "src/data/artificialtsunami_bathymetry_1000.nc";

  // create tsunami2d constructor
  tsunami_lab::setups::TsunamiEvent2d *l_tsunamiEvent2d = new tsunami_lab::setups::TsunamiEvent2d(
      l_displacement.data(),
      l_bathymetry.data(),
      &o_width,
      &o_height,
      &o_offsetX,
      &o_offsetY);

  tsunami_lab::setups::ArtificialTsunami2d *l_artificialTsunami2d = new tsunami_lab::setups::ArtificialTsunami2d();

  // check if the constructor works
  REQUIRE(o_width == Approx(9990));
  REQUIRE(o_height == Approx(9990));
  REQUIRE(o_offsetX == Approx(-4995));
  REQUIRE(o_offsetY == Approx(-4995));

  // check if artificialTsunami and tsunamiEvent2d are the same for the artificial tsunami data
  for (tsunami_lab::t_real l_x = -4985; l_x < 4985; l_x += 50)
    for (tsunami_lab::t_real l_y = -4985; l_y < 4985; l_y += 50)
    {
      REQUIRE(l_tsunamiEvent2d->getBathymetry(l_x, l_y) == Approx(l_artificialTsunami2d->getBathymetry(l_x, l_y)));
      REQUIRE(l_tsunamiEvent2d->getHeight(l_x, l_y) == Approx(l_artificialTsunami2d->getHeight(l_x, l_y)));
      REQUIRE(l_tsunamiEvent2d->getMomentumX(l_x, l_y) == Approx(l_artificialTsunami2d->getMomentumX(l_x, l_y)));
      REQUIRE(l_tsunamiEvent2d->getMomentumY(l_x, l_y) == Approx(l_artificialTsunami2d->getMomentumY(l_x, l_y)));
    }

  delete l_tsunamiEvent2d;
  delete l_artificialTsunami2d;
}