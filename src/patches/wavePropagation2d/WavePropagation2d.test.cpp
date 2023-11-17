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

TEST_CASE("Test the 2d wave propagation FWave solver x direction.", "[WaveProp2dFWaveX]")
{
    /*
     * Test case:
     *
     *   Single dam break problem between x-cell 49 and 50.
     *     left | right
     *       10 | 8
     *        0 | 0
     *
     *   Elsewhere steady state.
     *
     * The net-updates at the respective edge are given as
     * (see derivation in Roe solver):
     *    left          | right
     *      9.394671362 | -9.394671362
     *    -88.25985     | -88.25985
     *
     * Analogue to test of WavePropagation1d.test.cpp
     */

    tsunami_lab::patches::WavePropagation2d m_waveProp(100, 100, true, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

    for (std::size_t l_cx = 0; l_cx < 50; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 10);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }
    for (std::size_t l_cx = 50; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 8);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }

    // perform a time step
    m_waveProp.timeStep(0.1);

    // steady state
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(10));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // dam-break
    for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
    {
        REQUIRE(m_waveProp.getHeight()[(49 + 1) + (l_cy + 1) * 102] == Approx(10 - 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(49 + 1) + (l_cy + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getMomentumY()[(49 + 1) + (l_cy + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getBathymetry()[(49 + 1) + (l_cy + 1) * 102] == Approx(0));

        REQUIRE(m_waveProp.getHeight()[(50 + 1) + (l_cy + 1) * 102] == Approx(8 + 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (l_cy + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (l_cy + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (l_cy + 1) * 102] == Approx(0));
    }

    // steady state
    for (std::size_t l_cx = 51; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(8));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }
}

TEST_CASE("Test the 2d wave propagation FWave solver y direction.", "[WaveProp2dFWaveY]")
{
    /*
     * Test case:
     *
     *   Single dam break problem between y-cell 49 and 50.
     *     left | right
     *       10 | 8
     *        0 | 0
     *
     *   Elsewhere steady state.
     *
     * The net-updates at the respective edge are given as
     * (see derivation in Roe solver):
     *    left          | right
     *      9.394671362 | -9.394671362
     *    -88.25985     | -88.25985
     *
     * Analogue to test of WavePropagation1d.test.cpp
     */

    tsunami_lab::patches::WavePropagation2d m_waveProp(100, 100, true, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

    for (std::size_t l_cy = 0; l_cy < 50; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 10);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }
    for (std::size_t l_cy = 50; l_cy < 100; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 8);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }

    // perform a time step
    m_waveProp.timeStep(0.1);

    // steady state
    for (std::size_t l_cy = 0; l_cy < 49; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(10));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // dam-break
    for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
    {
        REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (49 + 1) * 102] == Approx(10 - 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (49 + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (49 + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (49 + 1) * 102] == Approx(0));

        REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (50 + 1) * 102] == Approx(8 + 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (50 + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (50 + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (50 + 1) * 102] == Approx(0));
    }

    // steady state
    for (std::size_t l_cy = 51; l_cy < 100; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(8));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }
}

TEST_CASE("Test the 2d wave propagation Roe solver x direction.", "[WaveProp2dRoeX]")
{
    /*
     * Test case:
     *
     *   Single dam break problem between x-cell 49 and 50.
     *     left | right
     *       10 | 8
     *        0 | 0
     *
     *   Elsewhere steady state.
     *
     * The net-updates at the respective edge are given as
     * (see derivation in Roe solver):
     *    left          | right
     *      9.394671362 | -9.394671362
     *    -88.25985     | -88.25985
     *
     * Analogue to test of WavePropagation1d.test.cpp
     */

    tsunami_lab::patches::WavePropagation2d m_waveProp(100, 100, false, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

    for (std::size_t l_cx = 0; l_cx < 50; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 10);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }
    for (std::size_t l_cx = 50; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 8);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }

    // perform a time step
    m_waveProp.timeStep(0.1);

    // steady state
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(10));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // dam-break
    for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
    {
        REQUIRE(m_waveProp.getHeight()[(49 + 1) + (l_cy + 1) * 102] == Approx(10 - 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(49 + 1) + (l_cy + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getMomentumY()[(49 + 1) + (l_cy + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getBathymetry()[(49 + 1) + (l_cy + 1) * 102] == Approx(0));

        REQUIRE(m_waveProp.getHeight()[(50 + 1) + (l_cy + 1) * 102] == Approx(8 + 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (l_cy + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (l_cy + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (l_cy + 1) * 102] == Approx(0));
    }

    // steady state
    for (std::size_t l_cx = 51; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(8));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }
}

TEST_CASE("Test the 2d wave propagation Roe solver y direction.", "[WaveProp2dRoeY]")
{
    /*
     * Test case:
     *
     *   Single dam break problem between y-cell 49 and 50.
     *     left | right
     *       10 | 8
     *        0 | 0
     *
     *   Elsewhere steady state.
     *
     * The net-updates at the respective edge are given as
     * (see derivation in Roe solver):
     *    left          | right
     *      9.394671362 | -9.394671362
     *    -88.25985     | -88.25985
     *
     * Analogue to test of WavePropagation1d.test.cpp
     */

    tsunami_lab::patches::WavePropagation2d m_waveProp(100, 100, false, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

    for (std::size_t l_cy = 0; l_cy < 50; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 10);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }
    for (std::size_t l_cy = 50; l_cy < 100; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 8);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }

    // perform a time step
    m_waveProp.timeStep(0.1);

    // steady state
    for (std::size_t l_cy = 0; l_cy < 49; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(10));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // dam-break
    for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
    {
        REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (49 + 1) * 102] == Approx(10 - 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (49 + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (49 + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (49 + 1) * 102] == Approx(0));

        REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (50 + 1) * 102] == Approx(8 + 0.1 * 9.394671362));
        REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (50 + 1) * 102] == Approx(0));
        REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (50 + 1) * 102] == Approx(0 + 0.1 * 88.25985));
        REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (50 + 1) * 102] == Approx(0));
    }

    // steady state
    for (std::size_t l_cy = 51; l_cy < 100; l_cy++)
        for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(8));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }
}

TEST_CASE("Test the 2d wave propagation FWave solver x direction.", "[WaveProp2dFWaveX]")
{
    /*
     * Test case:
     *
     *   Dam break problem between x-cell 49 and 50 in x and y direction.
     *     left upper quadrant | rest of grid
     *                      10 | 8
     *                       0 | 0
     *
     *   Elsewhere steady state.
     *
     * The net-updates at the respective edge are given as
     * (see derivation in Roe solver):
     *    left          | right
     *      9.394671362 | -9.394671362
     *    -88.25985     | -88.25985
     *
     * Analogue to test of WavePropagation1d.test.cpp
     */

    
    tsunami_lab::patches::WavePropagation2d m_waveProp(100, 100, true, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

    //allocate 50 x 50 cells with 10m height in a 100*100 grid
    for (std::size_t l_cx = 0; l_cx < 50; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 50; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 10);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }

    for (std::size_t l_cx = 0; l_cx < 50; l_cx++)
        for (std::size_t l_cy = 50; l_cy < 100; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 8);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }
        
    for (std::size_t l_cx = 50; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            m_waveProp.setHeight(l_cx,
                                 l_cy,
                                 8);
            m_waveProp.setMomentumX(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setMomentumY(l_cx,
                                    l_cy,
                                    0);
            m_waveProp.setBathymetry(l_cx,
                                     l_cy,
                                     0);
        }

    // perform a time step
    m_waveProp.timeStep(0.1);

    // steady state
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 49; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(10));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // dam-break x-direction
    for (std::size_t l_cx = 0; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 49; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(49 + 1) + (l_cx + 1) * 102] == Approx(10 - 0.1 * 9.394671362));
            REQUIRE(m_waveProp.getMomentumX()[(49 + 1) + (l_cx + 1) * 102] == Approx(0 + 0.1 * 88.25985));
            REQUIRE(m_waveProp.getMomentumY()[(49 + 1) + (l_cx + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(49 + 1) + (l_cx + 1) * 102] == Approx(0));

            REQUIRE(m_waveProp.getHeight()[(50 + 1) + (l_cx + 1) * 102] == Approx(8 + 0.1 * 9.394671362));
            REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (l_cx + 1) * 102] == Approx(0 + 0.1 * 88.25985));
            REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (l_cx + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (l_cx + 1) * 102] == Approx(0));
        }

    // dam-break y direction
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(49 + 1) + (l_cy + 1) * 102] == Approx(10 - 0.1 * 9.394671362));
            REQUIRE(m_waveProp.getMomentumX()[(49 + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(49 + 1) + (l_cy + 1) * 102] == Approx(0 + 0.1 * 88.25985));
            REQUIRE(m_waveProp.getBathymetry()[(49 + 1) + (l_cy + 1) * 102] == Approx(0));

            REQUIRE(m_waveProp.getHeight()[(50 + 1) + (l_cy + 1) * 102] == Approx(8 + 0.1 * 9.394671362));
            REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (l_cy + 1) * 102] == Approx(0 + 0.1 * 88.25985));
            REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // steady state
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
        for (std::size_t l_cy = 51; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(8));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // steady state
    for (std::size_t l_cx = 51; l_cx < 100; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 100; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(8));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }
    //since diagonal cells are not updated, they should still be zero
    REQUIRE(m_waveProp.getHeight()[50 + 50 * 102] == Approx(8));
    REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (50 + 1) * 102] == Approx(0));
    REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (50 + 1) * 102] == Approx(0));
    REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (50 + 1) * 102] == Approx(0));
}
