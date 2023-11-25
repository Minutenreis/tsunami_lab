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
#include <string>
#include <netcdf.h>
#include <ncCheck.h>

TEST_CASE("Test Writing NetCDF Files", "[NetCdfWrite]")
{
    tsunami_lab::io::IoWriter *l_writer = new tsunami_lab::io::NetCdf();
    tsunami_lab::t_real l_b[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    l_writer->init(1, 2, 2, 4, 1, 1, 0, 0, l_b);

    tsunami_lab::t_real l_h[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tsunami_lab::t_real l_hu[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tsunami_lab::t_real l_hv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    l_writer->write(l_h, l_hu, l_hv, 0.0, 0);

    tsunami_lab::t_real l_h2[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    tsunami_lab::t_real l_hu2[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    tsunami_lab::t_real l_hv2[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    l_writer->write(l_h2, l_hu2, l_hv2, 0.1, 1);

    delete l_writer;

    // check if file exists
    REQUIRE(std::filesystem::exists("output.nc"));

    int l_ncidp;
    // open netCDF file
    netCDF::ncCheck(nc_open("output.nc", NC_NOWRITE, &l_ncidp), __FILE__, __LINE__);

    // read dimensions
    int l_dimXId, l_dimYId, l_dimTimeId;
    netCDF::ncCheck(nc_inq_dimid(l_ncidp, "x", &l_dimXId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_dimid(l_ncidp, "y", &l_dimYId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_dimid(l_ncidp, "time", &l_dimTimeId), __FILE__, __LINE__);

    size_t l_nx, l_ny, l_nt;
    netCDF::ncCheck(nc_inq_dimlen(l_ncidp, l_dimXId, &l_nx), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_dimlen(l_ncidp, l_dimYId, &l_ny), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_dimlen(l_ncidp, l_dimTimeId, &l_nt), __FILE__, __LINE__);

    REQUIRE(l_nx == 2);
    REQUIRE(l_ny == 2);
    REQUIRE(l_nt == 2);

    // read variables
    int l_varXId, l_varYId, l_varTimeId, l_varBId, l_varHId, l_varHuId, l_varHvId;
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "x", &l_varXId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "y", &l_varYId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "time", &l_varTimeId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "bathymetry", &l_varBId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "height", &l_varHId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "momentum_x", &l_varHuId), __FILE__, __LINE__);
    netCDF::ncCheck(nc_inq_varid(l_ncidp, "momentum_y", &l_varHvId), __FILE__, __LINE__);

    tsunami_lab::t_real *l_xR = new tsunami_lab::t_real[2];
    tsunami_lab::t_real *l_yR = new tsunami_lab::t_real[2];
    tsunami_lab::t_real *l_timeR = new tsunami_lab::t_real[2];
    tsunami_lab::t_real *l_bR = new tsunami_lab::t_real[4];
    tsunami_lab::t_real *l_hR = new tsunami_lab::t_real[8];
    tsunami_lab::t_real *l_huR = new tsunami_lab::t_real[8];
    tsunami_lab::t_real *l_hvR = new tsunami_lab::t_real[8];

    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varXId, l_xR), __FILE__, __LINE__);
    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varYId, l_yR), __FILE__, __LINE__);
    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varTimeId, l_timeR), __FILE__, __LINE__);
    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varBId, l_bR), __FILE__, __LINE__);
    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varHId, l_hR), __FILE__, __LINE__);
    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varHuId, l_huR), __FILE__, __LINE__);
    netCDF::ncCheck(nc_get_var_float(l_ncidp, l_varHvId, l_hvR), __FILE__, __LINE__);

    REQUIRE(l_xR[0] == Approx(0.5));
    REQUIRE(l_xR[1] == Approx(1.5));
    REQUIRE(l_yR[0] == Approx(0.5));
    REQUIRE(l_yR[1] == Approx(1.5));
    REQUIRE(l_timeR[0] == Approx(0.0));
    REQUIRE(l_timeR[1] == Approx(0.1));
    REQUIRE(l_bR[0] == Approx(6));
    REQUIRE(l_bR[1] == Approx(7));
    REQUIRE(l_bR[2] == Approx(10));
    REQUIRE(l_bR[3] == Approx(11));
    REQUIRE(l_hR[0] == Approx(6));
    REQUIRE(l_hR[1] == Approx(7));
    REQUIRE(l_hR[2] == Approx(10));
    REQUIRE(l_hR[3] == Approx(11));
    REQUIRE(l_hR[4] == Approx(7));
    REQUIRE(l_hR[5] == Approx(8));
    REQUIRE(l_hR[6] == Approx(11));
    REQUIRE(l_hR[7] == Approx(12));
    REQUIRE(l_huR[0] == Approx(6));
    REQUIRE(l_huR[1] == Approx(7));
    REQUIRE(l_huR[2] == Approx(10));
    REQUIRE(l_huR[3] == Approx(11));
    REQUIRE(l_huR[4] == Approx(7));
    REQUIRE(l_huR[5] == Approx(8));
    REQUIRE(l_huR[6] == Approx(11));
    REQUIRE(l_huR[7] == Approx(12));
    REQUIRE(l_hvR[0] == Approx(6));
    REQUIRE(l_hvR[1] == Approx(7));
    REQUIRE(l_hvR[2] == Approx(10));
    REQUIRE(l_hvR[3] == Approx(11));
    REQUIRE(l_hvR[4] == Approx(7));
    REQUIRE(l_hvR[5] == Approx(8));
    REQUIRE(l_hvR[6] == Approx(11));
    REQUIRE(l_hvR[7] == Approx(12));

    delete[] l_xR;
    delete[] l_yR;
    delete[] l_timeR;
    delete[] l_bR;
    delete[] l_hR;
    delete[] l_huR;
    delete[] l_hvR;

    // close netCdf file
    netCDF::ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);
}

TEST_CASE("Test Reading NetCdf Data", "[NetCdfRead]")
{
    tsunami_lab::t_idx l_nx, l_ny;
    tsunami_lab::t_real *l_x, *l_y, *l_z;
    std::string l_file = "src/data/testDispl.nc";
    tsunami_lab::io::NetCdf::read(l_file.data(), &l_nx, &l_ny, &l_x, &l_y, &l_z);

    REQUIRE(l_nx == 5);
    REQUIRE(l_ny == 3);
    for (tsunami_lab::t_idx l_ix = 0; l_ix < l_nx; l_ix++)
        REQUIRE(l_x[l_ix] == l_ix + 3);
    for (tsunami_lab::t_idx l_iy = 0; l_iy < l_ny; l_iy++)
        REQUIRE(l_y[l_iy] == l_iy + 1);
    for (tsunami_lab::t_idx l_iz = 0; l_iz < l_ny * l_nx; l_iz++)
        REQUIRE(l_z[l_iz] == l_iz);

    delete[] l_x;
    delete[] l_y;
    delete[] l_z;
}