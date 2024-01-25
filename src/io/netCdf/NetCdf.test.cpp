/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * Test for the netCdf interface.
 **/
#include "NetCdf.h"
#include <catch2/catch.hpp>
#include "../IoWriter.h"
#include <filesystem>
#include <string>
#include <netcdf.h>

TEST_CASE("Test Writing NetCDF Files", "[NetCdfWrite]")
{
    tsunami_lab::t_real l_b[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tsunami_lab::io::IoWriter *l_writer = new tsunami_lab::io::NetCdf(1, 2, 2, 4, 1, 1, 0, 0, 1, l_b, false);

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
    tsunami_lab::io::NetCdf::ncCheck(nc_open("output.nc", NC_NOWRITE, &l_ncidp), __FILE__, __LINE__);

    // read dimensions
    int l_dimXId, l_dimYId, l_dimTimeId;
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimid(l_ncidp, "x", &l_dimXId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimid(l_ncidp, "y", &l_dimYId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimid(l_ncidp, "time", &l_dimTimeId), __FILE__, __LINE__);

    size_t l_nx, l_ny, l_nt;
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimlen(l_ncidp, l_dimXId, &l_nx), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimlen(l_ncidp, l_dimYId, &l_ny), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimlen(l_ncidp, l_dimTimeId, &l_nt), __FILE__, __LINE__);

    REQUIRE(l_nx == 2);
    REQUIRE(l_ny == 2);
    REQUIRE(l_nt == 2);

    // read variables
    int l_varXId, l_varYId, l_varTimeId, l_varBId, l_varHId, l_varHuId, l_varHvId;
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "x", &l_varXId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "y", &l_varYId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "time", &l_varTimeId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "bathymetry", &l_varBId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "height", &l_varHId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "momentum_x", &l_varHuId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "momentum_y", &l_varHvId), __FILE__, __LINE__);

    tsunami_lab::t_real *l_xR = new tsunami_lab::t_real[2];
    tsunami_lab::t_real *l_yR = new tsunami_lab::t_real[2];
    tsunami_lab::t_real *l_timeR = new tsunami_lab::t_real[2];
    tsunami_lab::t_real *l_bR = new tsunami_lab::t_real[4];
    tsunami_lab::t_real *l_hR = new tsunami_lab::t_real[8];
    tsunami_lab::t_real *l_huR = new tsunami_lab::t_real[8];
    tsunami_lab::t_real *l_hvR = new tsunami_lab::t_real[8];

    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varXId, l_xR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varYId, l_yR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varTimeId, l_timeR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varBId, l_bR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varHId, l_hR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varHuId, l_huR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varHvId, l_hvR), __FILE__, __LINE__);

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
    tsunami_lab::io::NetCdf::ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);

    // delete file
    std::filesystem::remove_all("output.nc");
}

TEST_CASE("Test Writing Coarse Output", "[NetCdfWriteCoarse]")
{
    tsunami_lab::t_real l_b[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    tsunami_lab::io::IoWriter *l_writer = new tsunami_lab::io::NetCdf(1, 4, 4, 4, 0, 0, 0, -1, 3, l_b, false);

    // check if file exists
    REQUIRE(std::filesystem::exists("output.nc"));

    int l_ncidp;
    // open netCDF file
    tsunami_lab::io::NetCdf::ncCheck(nc_open("output.nc", NC_NOWRITE, &l_ncidp), __FILE__, __LINE__);

    // read dimensions
    int l_dimXId, l_dimYId, l_dimTimeId;
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimid(l_ncidp, "x", &l_dimXId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimid(l_ncidp, "y", &l_dimYId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimid(l_ncidp, "time", &l_dimTimeId), __FILE__, __LINE__);

    size_t l_nx, l_ny, l_nt;
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimlen(l_ncidp, l_dimXId, &l_nx), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimlen(l_ncidp, l_dimYId, &l_ny), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_dimlen(l_ncidp, l_dimTimeId, &l_nt), __FILE__, __LINE__);

    REQUIRE(l_nx == 1);
    REQUIRE(l_ny == 1);
    REQUIRE(l_nt == 0);

    // read variables
    int l_varXId, l_varYId, l_varBId;

    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "x", &l_varXId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "y", &l_varYId), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_inq_varid(l_ncidp, "bathymetry", &l_varBId), __FILE__, __LINE__);

    tsunami_lab::t_real l_xR, l_yR, l_bR;

    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varXId, &l_xR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varYId, &l_yR), __FILE__, __LINE__);
    tsunami_lab::io::NetCdf::ncCheck(nc_get_var_float(l_ncidp, l_varBId, &l_bR), __FILE__, __LINE__);

    REQUIRE(l_xR == Approx(1.5));
    REQUIRE(l_yR == Approx(0.5));
    REQUIRE(l_bR == Approx(5));

    delete l_writer;

    // delete file
    std::filesystem::remove_all("output.nc");
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

TEST_CASE("Test Checkpointing", "[NetCdfCheckpoints]")
{
    tsunami_lab::t_idx l_nx = 2,
                       l_ny = 2,
                       l_stride = l_nx,
                       l_ghostCellsX = 0,
                       l_ghostCellsY = 0;

    bool l_useFWave = true;
    bool l_useCuda = false;

    tsunami_lab::t_boundary l_boundaryL = tsunami_lab::t_boundary::OPEN,
                            l_boundaryR = tsunami_lab::t_boundary::WALL,
                            l_boundaryB = tsunami_lab::t_boundary::OPEN,
                            l_boundaryT = tsunami_lab::t_boundary::WALL;

    tsunami_lab::t_real l_endtime = 2.3,
                        l_width = 1.1,
                        l_xOffset = 0.1,
                        l_yOffset = 0.2,
                        l_hMax = 0.5,
                        *l_b = new tsunami_lab::t_real[l_nx * l_ny],
                        *l_h = new tsunami_lab::t_real[l_nx * l_ny],
                        *l_hu = new tsunami_lab::t_real[l_nx * l_ny],
                        *l_hv = new tsunami_lab::t_real[l_nx * l_ny];

    std::string l_stationFilePath = "src/data/stations.txt";

    tsunami_lab::t_idx l_nFrames = 2,
                       l_k = 1,
                       l_timeStep = 5,
                       l_nOut = 3,
                       l_nFreqStation = 4;

    tsunami_lab::t_real l_simTime = 0.3;

    int l_maxHours = 2;

    for (tsunami_lab::t_idx l_ix = 0; l_ix < l_nx * l_ny; l_ix++)
    {
        l_b[l_ix] = l_ix;
        l_h[l_ix] = l_ix;
        l_hu[l_ix] = l_ix;
        l_hv[l_ix] = l_ix;
    }

    tsunami_lab::io::NetCdf::writeCheckpoint(l_nx,
                                             l_ny,
                                             l_stride,
                                             l_ghostCellsX,
                                             l_ghostCellsY,
                                             l_useFWave,
                                             l_useCuda,
                                             l_boundaryL,
                                             l_boundaryR,
                                             l_boundaryB,
                                             l_boundaryT,
                                             l_endtime,
                                             l_width,
                                             l_xOffset,
                                             l_yOffset,
                                             l_hMax,
                                             l_stationFilePath,
                                             l_nFrames,
                                             l_k,
                                             l_timeStep,
                                             l_nOut,
                                             l_nFreqStation,
                                             l_simTime,
                                             l_maxHours,
                                             l_b,
                                             l_h,
                                             l_hu,
                                             l_hv);

    // check if directory exists
    REQUIRE(std::filesystem::exists("checkpoints"));

    // find newest checkpoint
    std::vector<std::string> l_checkpoints = {};
    for (const auto &entry : std::filesystem::directory_iterator("checkpoints"))
    {
        l_checkpoints.push_back(entry.path());
    }
    std::sort(l_checkpoints.begin(), l_checkpoints.end());
    std::string l_newestCheckpoint = l_checkpoints.back();

    // check all values
    tsunami_lab::t_idx l_nxR,
        l_nyR;

    bool l_useFWaveR;
    bool l_useCudaR;

    tsunami_lab::t_boundary l_boundaryLR,
        l_boundaryRR,
        l_boundaryBR,
        l_boundaryTR;

    tsunami_lab::t_real l_endtimeR,
        l_widthR,
        l_xOffsetR,
        l_yOffsetR,
        l_hMaxR,
        *l_bR,
        *l_hR,
        *l_huR,
        *l_hvR;

    std::string l_stationFilePathR;

    tsunami_lab::t_idx l_nFramesR,
        l_kR,
        l_timeStepR,
        l_nOutR,
        l_nFreqStationR;

    tsunami_lab::t_real l_simTimeR;

    int l_maxHoursR;

    tsunami_lab::io::NetCdf::readCheckpoint(l_newestCheckpoint.data(),
                                            &l_nxR,
                                            &l_nyR,
                                            &l_useFWaveR,
                                            &l_useCudaR,
                                            &l_boundaryLR,
                                            &l_boundaryRR,
                                            &l_boundaryBR,
                                            &l_boundaryTR,
                                            &l_endtimeR,
                                            &l_widthR,
                                            &l_xOffsetR,
                                            &l_yOffsetR,
                                            &l_hMaxR,
                                            &l_stationFilePathR,
                                            &l_nFramesR,
                                            &l_kR,
                                            &l_timeStepR,
                                            &l_nOutR,
                                            &l_nFreqStationR,
                                            &l_simTimeR,
                                            &l_maxHoursR,
                                            &l_bR,
                                            &l_hR,
                                            &l_huR,
                                            &l_hvR);

    REQUIRE(l_nxR == l_nx);
    REQUIRE(l_nyR == l_ny);
    REQUIRE(l_useFWaveR == l_useFWave);
    REQUIRE(l_useCudaR == l_useCuda);
    REQUIRE(l_boundaryLR == l_boundaryL);
    REQUIRE(l_boundaryRR == l_boundaryR);
    REQUIRE(l_boundaryBR == l_boundaryB);
    REQUIRE(l_boundaryTR == l_boundaryT);
    REQUIRE(l_endtimeR == l_endtime);
    REQUIRE(l_widthR == l_width);
    REQUIRE(l_xOffsetR == l_xOffset);
    REQUIRE(l_yOffsetR == l_yOffset);
    REQUIRE(l_hMaxR == l_hMax);
    REQUIRE(l_stationFilePathR == l_stationFilePath);
    REQUIRE(l_nFramesR == l_nFrames);
    REQUIRE(l_kR == l_k);
    REQUIRE(l_timeStepR == l_timeStep);
    REQUIRE(l_nOutR == l_nOut);
    REQUIRE(l_nFreqStationR == l_nFreqStation);
    REQUIRE(l_simTimeR == l_simTime);
    REQUIRE(l_maxHoursR == l_maxHours);

    for (tsunami_lab::t_idx l_ix = 0; l_ix < l_nx * l_ny; l_ix++)
    {
        REQUIRE(l_bR[l_ix] == l_b[l_ix]);
        REQUIRE(l_hR[l_ix] == l_h[l_ix]);
        REQUIRE(l_huR[l_ix] == l_hu[l_ix]);
        REQUIRE(l_hvR[l_ix] == l_hv[l_ix]);
    }

    delete[] l_b;
    delete[] l_h;
    delete[] l_hu;
    delete[] l_hv;
    delete[] l_bR;
    delete[] l_hR;
    delete[] l_huR;
    delete[] l_hvR;

    // delete checkpoint
    std::filesystem::remove_all("checkpoints");
}