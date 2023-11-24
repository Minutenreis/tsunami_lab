/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * Test for the netCdf interface.
 **/
#include <catch2/catch.hpp>
#include "NetCdf.h"
#include "../IoWriter.h"
#include <filesystem>

TEST_CASE("Test Writing NetCDF Files", "[NetCdfWrite]")
{
    tsunami_lab::io::IoWriter *l_writer = new tsunami_lab::io::NetCdf();
    tsunami_lab::t_real l_b[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    l_writer->init(1, 2, 2, 4, 1, 1, 0, 0, l_b);

    tsunami_lab::t_real l_h[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tsunami_lab::t_real l_hu[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tsunami_lab::t_real l_hv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    l_writer->write(l_h, l_hu, l_hv, 0, 0.0);

    tsunami_lab::t_real l_h2[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    tsunami_lab::t_real l_hu2[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    tsunami_lab::t_real l_hv2[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    l_writer->write(l_h2, l_hu2, l_hv2, 1, 0.1);

    delete l_writer;

    // check if file exists
    REQUIRE(std::filesystem::exists("output.nc"));

    // todo: read data back in and compare
}