/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Unit tests for the two-dimensional wave propagation patch.
 **/
#include <catch2/catch.hpp>
#include "WavePropagation2d.h"
#include "../../constants.h"


TEST_CASE("Test the 2d wave propagation FWave solver.", "[WaveProp2dFWave]")
{
  tsunami_lab::patches::WavePropagation2d m_waveProp(100, 100, true,  tsunami_lab::t_boundary::OPEN,  tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

  for (std::size_t l_ce = 0; l_ce < 50; l_ce++)
  {
    m_waveProp.setHeight(l_ce,
                         0,
                         10);
    m_waveProp.setMomentumX(l_ce,
                            0,
                            0);
    m_waveProp.setBathymetry(l_ce,
                             0,
                             0);
  }
  for (std::size_t l_ce = 50; l_ce < 100; l_ce++)
  {
    m_waveProp.setHeight(l_ce,
                         0,
                         8);
    m_waveProp.setMomentumX(l_ce,
                            0,
                            0);
    m_waveProp.setBathymetry(l_ce,
                             0,
                             0);
  }

  // perform a time step
  m_waveProp.timeStep(0.1);

  // steady state
  for (std::size_t l_ce = 0; l_ce < 49; l_ce++)
  {
    REQUIRE(m_waveProp.getHeight()[l_ce + 1] == Approx(10));
    REQUIRE(m_waveProp.getMomentumX()[l_ce + 1] == Approx(0));
  }

  // dam-break
  REQUIRE(m_waveProp.getHeight()[49 + 1] == Approx(10.0f));
  REQUIRE(m_waveProp.getMomentumX()[49 + 1] == Approx(0.0f));

  REQUIRE(m_waveProp.getHeight()[50 + 1] == Approx(8.0f));
  REQUIRE(m_waveProp.getMomentumX()[50 + 1] == Approx(0.0f));

  // steady state
  for (std::size_t l_ce = 51; l_ce < 100; l_ce++)
  {
    REQUIRE(m_waveProp.getHeight()[l_ce + 1] == Approx(8));
    REQUIRE(m_waveProp.getMomentumX()[l_ce + 1] == Approx(0));
  }
}