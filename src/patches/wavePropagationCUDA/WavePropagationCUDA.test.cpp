/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Unit tests for the two-dimensional wave propagation patch.
 **/
#include <catch2/catch.hpp>
#include "WavePropagationCUDA.h"
#include "../../constants.h"

TEST_CASE("Test the CUDA wave propagation solver x direction.", "[CUDA]")
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

    tsunami_lab::patches::WavePropagationCUDA m_waveProp(100, 100, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

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
    m_waveProp.initGhostCells();

    // perform a time step
    m_waveProp.timeStep(0.1);

    m_waveProp.prepareDataAccess();

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

TEST_CASE("Test the CUDA wave propagation solver y direction.", "[CUDA]")
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

    tsunami_lab::patches::WavePropagationCUDA m_waveProp(100, 100, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

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
    m_waveProp.initGhostCells();

    // perform a time step
    m_waveProp.timeStep(0.1);

    m_waveProp.prepareDataAccess();

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

TEST_CASE("Test the 2d wave propagation solver diagonally.", "[CUDA]")
{
    /*
     * Test case:
     *
     *   Dam break problem between x-cell 49 and 50 in for y e [0,49] and y cells for x e [0,49].
     *     left lower quadrant | rest of grid
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

    tsunami_lab::patches::WavePropagationCUDA m_waveProp(100, 100, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN, tsunami_lab::t_boundary::OPEN);

    // allocate 50 x 50 cells with 10m height in a 100*100 grid
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
    m_waveProp.initGhostCells();

    // perform a time step
    m_waveProp.timeStep(0.1);

    m_waveProp.prepareDataAccess();

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
    for (std::size_t l_cy = 0; l_cy < 49; l_cy++)
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

    // dam-break y direction
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
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
    for (std::size_t l_cx = 0; l_cx < 49; l_cx++)
        for (std::size_t l_cy = 0; l_cy < 49; l_cy++)
        {
            REQUIRE(m_waveProp.getHeight()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(10));
            REQUIRE(m_waveProp.getMomentumX()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getMomentumY()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
            REQUIRE(m_waveProp.getBathymetry()[(l_cx + 1) + (l_cy + 1) * 102] == Approx(0));
        }

    // steady state
    for (std::size_t l_cx = 0; l_cx < 51; l_cx++)
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

    // 2d updates in diagonal cells
    // updated from 2 sides, bottom cell
    /**
     * the x-sweep is equal to the 1d case
     *
     * after: x-momentum = 0.1 * 88.25985
     *        height = 10 - 0.1 * 9.394671362
     *
     * y-sweep:
     *
     * top  | bottom
     *  8   | 10 - 0.1 * 9.394671362
     *  0   | 0
     * => y momentum still 0
     *
     * FWave speeds are given as:
     *  s1 = -sqrt(9.80665 * (9 - 0.05 * 9.394671362))
     *  s2 = sqrt(9.80665 * (9 - 0.05 * 9.394671362))
     *
     * Inversion of the matrix of right Eigenvectors:
     *
     *  wolframalpha.com query: invert {{1, 1}, {-sqrt(9.80665 * (9 - 0.05 * 9.394671362)), sqrt(9.80665 * (9 - 0.05 * 9.394671362))}}
     *
     *         | 0.5 -0.0546674 |
     *  Rinv = |                |
     *         | 0.5 0.0546674  |
     *
     * Multiplicaton with the jump in fluxes gives the wave strengths:
     *
     *         |  0 - 0                                                 |   |  4.84993 |   | a1 |
     *  Rinv * |                                                        | = |          | = |    |
     *         | 1/2*9.80665*8^2-1/2*9.80665*(10 - 0.1 * 9.394671362)^2 |   | -4.84993 |   | a2 |
     *
     * The net-updates are given through the scaled eigenvectors.
     *
     *                       |  1 |   | 4.84993       |
     *  update #1:      a1 * |    | = |               |
     *                       | s1 |   | -44.3585      |
     *
     *                       |  1 |   | -4.84993      |
     *  update #2:      a2 * |    | = |               |
     *                       | s2 |   | -44.3585      |
     *
     * bottom cell:
     * height = 10 - 0.1 * 9.394671362 - 0.1 * 4.84993
     * x-momentum = 0.1 * 88.25985
     * y-momentum = 0.1 * 44.3585
     */
    REQUIRE(m_waveProp.getHeight()[(49 + 1) + (49 + 1) * 102] == Approx(10 - 0.1 * 9.394671362 - 0.1 * 4.84993));
    REQUIRE(m_waveProp.getMomentumX()[(49 + 1) + (49 + 1) * 102] == Approx(0.1 * 88.25985));
    REQUIRE(m_waveProp.getMomentumY()[(49 + 1) + (49 + 1) * 102] == Approx(0.1 * 44.3585));
    REQUIRE(m_waveProp.getBathymetry()[(49 + 1) + (49 + 1) * 102] == Approx(0));

    // updated from 2 sides, top cell
    /**
     * the x-sweep is equal to the 1d case
     *
     * after bottom cell: x-momentum = 0.1 * 88.25985
     *                    height = 8 + 0.1 * 9.394671362
     *
     * y-sweep:
     *
     * top  | bottom
     *  8   | 8 + 0.1 * 9.394671362
     *  0   | 0
     * => y momentum still 0
     *
     * FWave speeds are given as:
     *  s1 = -sqrt(9.80665 * (8 + 0.05 * 9.394671362))
     *  s2 = sqrt(9.80665 * (8 + 0.05 * 9.394671362))
     *
     * Inversion of the matrix of right Eigenvectors:
     *
     *  wolframalpha.com query: invert {{1, 1}, {-sqrt(9.80665 * (8 + 0.05 * 9.394671362)), sqrt(9.80665 * (8 + 0.05 * 9.394671362))}}
     *
     *         | 0.5 -0.0548624 |
     *  Rinv = |                |
     *         | 0.5 0.0548624  |
     *
     * Multiplicaton with the jump in fluxes gives the wave strengths:
     *
     *         |  0 - 0                                                 |   |  4.28102 |   | a1 |
     *  Rinv * |                                                        | = |          | = |    |
     *         | 1/2*9.80665*8^2-1/2*9.80665*(8 + 0.1 * 9.394671362)^2  |   | -4.28102 |   | a2 |
     *
     * The net-updates are given through the scaled eigenvectors.
     *
     *                       |  1 |   | 4.28102       |
     *  update #1:      a1 * |    | = |               |
     *                       | s1 |   | -39.0160      |
     *
     *                       |  1 |   | -4.28102      |
     *  update #2:      a2 * |    | = |               |
     *                       | s2 |   | -39.0160      |
     *
     * top cell:
     * height = 8 + 0.1 * 4.28102
     * x-momentum = 0
     * y-momentum = 0.1 * 39.0160
     */
    REQUIRE(m_waveProp.getHeight()[(50 + 1) + (50 + 1) * 102] == Approx(8 + 0.1 * 4.28102));
    REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (50 + 1) * 102] == Approx(0));
    REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (50 + 1) * 102] == Approx(0.1 * 39.0160));
    REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (50 + 1) * 102] == Approx(0));

    // top cell from the calculation in (49,49) above
    // h = 8 + 0.1 * 4.84993
    // hu = 0
    // hv = 0.1 * 44.3585
    REQUIRE(m_waveProp.getHeight()[(49 + 1) + (50 + 1) * 102] == Approx(8 + 0.1 * 4.84993));
    REQUIRE(m_waveProp.getMomentumX()[(49 + 1) + (50 + 1) * 102] == Approx(0));
    REQUIRE(m_waveProp.getMomentumY()[(49 + 1) + (50 + 1) * 102] == Approx(0 + 0.1 * 44.3585));
    REQUIRE(m_waveProp.getBathymetry()[(49 + 1) + (50 + 1) * 102] == Approx(0));

    // bottom cell from the calculation in (50,50) above
    // h = 8 + 0.1 * 9.394671362 - 0.1 * 4.28102
    // hu = 0.1 * 88.25985
    // hv = 0.1 * 39.0160
    REQUIRE(m_waveProp.getHeight()[(50 + 1) + (49 + 1) * 102] == Approx(8 + 0.1 * 9.394671362 - 0.1 * 4.28102));
    REQUIRE(m_waveProp.getMomentumX()[(50 + 1) + (49 + 1) * 102] == Approx(0 + 0.1 * 88.25985));
    REQUIRE(m_waveProp.getMomentumY()[(50 + 1) + (49 + 1) * 102] == Approx(0.1 * 39.0160));
    REQUIRE(m_waveProp.getBathymetry()[(50 + 1) + (49 + 1) * 102] == Approx(0));
}
