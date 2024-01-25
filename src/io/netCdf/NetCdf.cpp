/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * IO-routines for reading / writing netCdf data.
 **/
#include <netcdf.h>
#include "NetCdf.h"
#include <iostream>
#include <string.h>
#include <filesystem>

void tsunami_lab::io::NetCdf::ncCheck(int i_status, char const *i_file, int i_line)
{
    if (i_status != NC_NOERR)
    {
        std::cerr << "NetCdf Error: " << nc_strerror(i_status) << " in " << i_file << " at line " << i_line << std::endl;
        exit(EXIT_FAILURE);
    }
}

void tsunami_lab::io::NetCdf::putVaraWithGhostcells(t_real const *i_data, int l_ncidp, int i_var, t_idx i_nOut, bool i_hasTime)
{
    t_idx l_time = i_hasTime ? 0 : 1; // if it has no time, start array at 1st index

    if (m_k == 1)
    {
        // if k == 1, no averaging is needed
        t_idx start_p[3] = {i_nOut, 0, 0};
        t_idx count_p[3] = {1, 1, m_nx};
        // for (start_p[1] = 0; start_p[1] < m_ny; ++start_p[1])
        // {
        //     ncCheck(nc_put_vara_float(l_ncidp, i_var, start_p + l_time, count_p + l_time, i_data + (start_p[1] + m_ghostCellsY) * m_stride + m_ghostCellsX), __FILE__, __LINE__);
        // }
        ptrdiff_t l_stride[3] = {1, 1, 1};
        ptrdiff_t l_imapp[3] = {1, (ptrdiff_t)m_stride, 1};
        ncCheck(nc_put_varm_float(l_ncidp, i_var, start_p + l_time, count_p + l_time, l_stride + l_time, l_imapp + l_time, i_data), __FILE__, __LINE__);
        return;
    }
    t_idx start_p[3] = {i_nOut, 0, 0};
    t_idx count_p[3] = {1, 1, m_nx / m_k};
    t_idx l_sizeX = (m_nx / m_k) * m_k; // m_nx/k*k (integer division) => ignores the overstanding cells at the right border
    t_real l_kSquaredInv = 1.0 / (m_k * m_k);
    for (start_p[1] = 0; start_p[1] < m_ny / m_k; ++start_p[1])
    {
        // zero initialised array for averaged data
        t_real *l_row = new t_real[m_nx / m_k]{};
        for (t_idx l_iy = start_p[1] * m_k; l_iy < (start_p[1] + 1) * m_k; ++l_iy)
        {
#pragma omp parallel for schedule(static, m_k)
            for (t_idx l_ix = 0; l_ix < l_sizeX; ++l_ix)
            {
                l_row[l_ix / m_k] += i_data[l_ix + m_ghostCellsX + (l_iy + m_ghostCellsY) * m_stride];
            }
        }
#pragma omp parallel for
        for (t_idx l_ix = 0; l_ix < m_nx / m_k; ++l_ix)
        {
            l_row[l_ix] *= l_kSquaredInv;
        }
        ncCheck(nc_put_vara_float(l_ncidp, i_var, start_p + l_time, count_p + l_time, l_row), __FILE__, __LINE__);
        delete[] l_row;
    }
}

tsunami_lab::io::NetCdf::NetCdf(t_real i_dxy,
                                t_idx i_nx,
                                t_idx i_ny,
                                t_idx i_stride,
                                t_idx i_ghostCellsX,
                                t_idx i_ghostCellsY,
                                t_real i_offsetX,
                                t_real i_offsetY,
                                t_real i_k,
                                t_real const *i_b,
                                bool i_useCheckpoint) : m_dxy(i_dxy),
                                                        m_nx(i_nx),
                                                        m_ny(i_ny),
                                                        m_stride(i_stride),
                                                        m_ghostCellsX(i_ghostCellsX),
                                                        m_ghostCellsY(i_ghostCellsY),
                                                        m_offsetX(i_offsetX),
                                                        m_offsetY(i_offsetY),
                                                        m_k(i_k)
{
    // if checkpoint is used, the file already exists with all static data
    if (!i_useCheckpoint)
    {
        int l_ncidp = -1;

        // create netCdf file
        ncCheck(nc_create("output.nc", NC_CLOBBER | NC_NETCDF4, &l_ncidp), __FILE__, __LINE__);

        // define dimensions & variables
        int l_dimXId, l_dimYId, l_dimTimeId;
        ncCheck(nc_def_dim(l_ncidp, "x", m_nx / m_k, &l_dimXId), __FILE__, __LINE__);
        ncCheck(nc_def_dim(l_ncidp, "y", m_ny / m_k, &l_dimYId), __FILE__, __LINE__);
        ncCheck(nc_def_dim(l_ncidp, "time", NC_UNLIMITED, &l_dimTimeId), __FILE__, __LINE__);

        int l_varXId, l_varYId, l_varTimeId, l_varHId, l_varHuId, l_varHvId, l_varBId;

        int l_dimB[2] = {l_dimYId, l_dimXId};
        int l_dimQ[3] = {l_dimTimeId, l_dimYId, l_dimXId};
        ncCheck(nc_def_var(l_ncidp, "x", NC_FLOAT, 1, &l_dimXId, &l_varXId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(l_ncidp, l_dimXId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);
        ncCheck(nc_def_var(l_ncidp, "y", NC_FLOAT, 1, &l_dimYId, &l_varYId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(l_ncidp, l_dimYId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);
        ncCheck(nc_def_var(l_ncidp, "time", NC_FLOAT, 1, &l_dimTimeId, &l_varTimeId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(l_ncidp, l_dimTimeId, "units", strlen("seconds since simulationstart"), "seconds since simulationstart"), __FILE__, __LINE__);

        ncCheck(nc_def_var(l_ncidp, "height", NC_FLOAT, 3, l_dimQ, &l_varHId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(l_ncidp, l_varHId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);
        ncCheck(nc_def_var(l_ncidp, "momentum_x", NC_FLOAT, 3, l_dimQ, &l_varHuId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(l_ncidp, l_varHuId, "units", strlen("newton second"), "newton second"), __FILE__, __LINE__);
        if (m_ny > 1)
        {
            ncCheck(nc_def_var(l_ncidp, "momentum_y", NC_FLOAT, 3, l_dimQ, &l_varHvId), __FILE__, __LINE__);
            ncCheck(nc_put_att_text(l_ncidp, l_varHvId, "units", strlen("newton second"), "newton second"), __FILE__, __LINE__);
        }
        ncCheck(nc_def_var(l_ncidp, "bathymetry", NC_FLOAT, 2, l_dimB, &l_varBId), __FILE__, __LINE__);
        ncCheck(nc_put_att_text(l_ncidp, l_varBId, "units", strlen("meter"), "meter"), __FILE__, __LINE__);

        // write data
        ncCheck(nc_enddef(l_ncidp), __FILE__, __LINE__);

        // generate x and y dimensions
        t_real *l_x = new t_real[m_nx / m_k];
        t_real *l_y = new t_real[m_ny / m_k];
        for (t_idx l_ix = 0; l_ix < m_nx / m_k; l_ix++)
        {
            l_x[l_ix] = m_offsetX + ((float)l_ix + 0.5) * (float)m_k * m_dxy;
        }
        for (t_idx l_iy = 0; l_iy < m_ny / m_k; l_iy++)
        {
            l_y[l_iy] = m_offsetY + ((float)l_iy + 0.5) * (float)m_k * m_dxy;
        }
        ncCheck(nc_put_var_float(l_ncidp, l_varXId, l_x), __FILE__, __LINE__);
        ncCheck(nc_put_var_float(l_ncidp, l_varYId, l_y), __FILE__, __LINE__);

        // write bathymetry
        putVaraWithGhostcells(i_b, l_ncidp, l_varBId, 0, false);
        delete[] l_x;
        delete[] l_y;
        ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);
    }
}

void tsunami_lab::io::NetCdf::write(t_real const *i_h,
                                    t_real const *i_hu,
                                    t_real const *i_hv,
                                    t_real i_time,
                                    t_idx i_nOut)
{
    int l_ncidp = -1;
    ncCheck(nc_open("output.nc", NC_WRITE, &l_ncidp), __FILE__, __LINE__);
    if (l_ncidp == -1)
    {
        std::cerr << "NetCdf Error: File not initialized!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // get variable ID's
    int l_varHId, l_varHuId, l_varHvId, l_varTimeId;
    ncCheck(nc_inq_varid(l_ncidp, "height", &l_varHId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "momentum_x", &l_varHuId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "time", &l_varTimeId), __FILE__, __LINE__);

    // write data
    putVaraWithGhostcells(i_h, l_ncidp, l_varHId, i_nOut, true);
    putVaraWithGhostcells(i_hu, l_ncidp, l_varHuId, i_nOut, true);
    // write momentum_y only if ny > 1 (2D)
    if (m_ny > 1)
    {
        ncCheck(nc_inq_varid(l_ncidp, "momentum_y", &l_varHvId), __FILE__, __LINE__);
        putVaraWithGhostcells(i_hv, l_ncidp, l_varHvId, i_nOut, true);
    }
    // write time
    ncCheck(nc_put_var1_float(l_ncidp, l_varTimeId, &i_nOut, &i_time), __FILE__, __LINE__);
    ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);
}

void tsunami_lab::io::NetCdf::read(char *i_filePath,
                                   t_idx *o_nx,
                                   t_idx *o_ny,
                                   t_real **o_x,
                                   t_real **o_y,
                                   t_real **o_z)
{
    int l_ncidp = -1;
    // open netCdf file
    ncCheck(nc_open(i_filePath, NC_NOWRITE, &l_ncidp), __FILE__, __LINE__);

    // read dimensions
    int l_dimXId, l_dimYId;
    ncCheck(nc_inq_dimid(l_ncidp, "x", &l_dimXId), __FILE__, __LINE__);
    ncCheck(nc_inq_dimid(l_ncidp, "y", &l_dimYId), __FILE__, __LINE__);

    // get dimension lengths
    ncCheck(nc_inq_dimlen(l_ncidp, l_dimXId, o_nx), __FILE__, __LINE__);
    ncCheck(nc_inq_dimlen(l_ncidp, l_dimYId, o_ny), __FILE__, __LINE__);

    // read variables
    int l_varXId, l_varYId, l_varZId;
    ncCheck(nc_inq_varid(l_ncidp, "x", &l_varXId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "y", &l_varYId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "z", &l_varZId), __FILE__, __LINE__);

    *o_x = new t_real[*o_nx];
    *o_y = new t_real[*o_ny];
    *o_z = new t_real[(*o_nx) * (*o_ny)];

    ncCheck(nc_get_var_float(l_ncidp, l_varXId, *o_x), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varYId, *o_y), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varZId, *o_z), __FILE__, __LINE__);

    // close netCdf file
    ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);
}

void tsunami_lab::io::NetCdf::readCheckpoint(char *i_filePath,
                                             t_idx *o_nx,
                                             t_idx *o_ny,
                                             bool *o_useFWave,
                                             bool *o_useCuda,
                                             tsunami_lab::t_boundary *o_boundaryL,
                                             tsunami_lab::t_boundary *o_boundaryR,
                                             tsunami_lab::t_boundary *o_boundaryB,
                                             tsunami_lab::t_boundary *o_boundaryT,
                                             tsunami_lab::t_real *o_endTime,
                                             tsunami_lab::t_real *o_width,
                                             tsunami_lab::t_real *o_xOffset,
                                             tsunami_lab::t_real *o_yOffset,
                                             tsunami_lab::t_real *o_hMax,
                                             std::string *o_stationFilePath,
                                             tsunami_lab::t_idx *o_nFrames,
                                             tsunami_lab::t_idx *o_k,
                                             tsunami_lab::t_idx *o_timeStep,
                                             tsunami_lab::t_idx *o_nOut,
                                             tsunami_lab::t_idx *o_nFreqStation,
                                             tsunami_lab::t_real *o_simTime,
                                             int *o_maxHours,
                                             t_real **o_b,
                                             t_real **o_h,
                                             t_real **o_hu,
                                             t_real **o_hv)
{
    // open netCdf file
    int l_ncidp = -1;
    ncCheck(nc_open(i_filePath, NC_NOWRITE, &l_ncidp), __FILE__, __LINE__);

    // read dimensions
    int l_dimXId, l_dimYId, l_dimTextId;
    ncCheck(nc_inq_dimid(l_ncidp, "x", &l_dimXId), __FILE__, __LINE__);
    ncCheck(nc_inq_dimid(l_ncidp, "y", &l_dimYId), __FILE__, __LINE__);
    ncCheck(nc_inq_dimid(l_ncidp, "text", &l_dimTextId), __FILE__, __LINE__);

    // get dimension lengths
    t_idx l_textLength;
    ncCheck(nc_inq_dimlen(l_ncidp, l_dimXId, o_nx), __FILE__, __LINE__);
    ncCheck(nc_inq_dimlen(l_ncidp, l_dimYId, o_ny), __FILE__, __LINE__);
    ncCheck(nc_inq_dimlen(l_ncidp, l_dimTextId, &l_textLength), __FILE__, __LINE__);

    // read dimensionless variables
    int l_varUseFWaveId, l_varUseCudaId, l_varBoundaryLId, l_varBoundaryRId, l_varBoundaryBId, l_varBoundaryTId;
    int l_varEndTimeId, l_varWidthId, l_varXOffsetId, l_varYOffsetId, l_varHMaxId;
    int l_varStationFilePathId, l_varNFramesId, l_varKId, l_varTimeStepId, l_varNOutId, l_varNFreqStationId;
    int l_varSimTimeId, l_varMaxHoursId;
    ncCheck(nc_inq_varid(l_ncidp, "useFWave", &l_varUseFWaveId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "useCuda", &l_varUseCudaId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "boundaryL", &l_varBoundaryLId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "boundaryR", &l_varBoundaryRId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "boundaryB", &l_varBoundaryBId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "boundaryT", &l_varBoundaryTId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "endTime", &l_varEndTimeId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "width", &l_varWidthId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "xOffset", &l_varXOffsetId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "yOffset", &l_varYOffsetId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "hMax", &l_varHMaxId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "stationFilePath", &l_varStationFilePathId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "nFrames", &l_varNFramesId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "k", &l_varKId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "timeStep", &l_varTimeStepId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "nOut", &l_varNOutId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "nFreqStation", &l_varNFreqStationId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "simTime", &l_varSimTimeId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "maxHours", &l_varMaxHoursId), __FILE__, __LINE__);

    // read array id's
    int l_varBId, l_varHId, l_varHuId, l_varHvId;
    ncCheck(nc_inq_varid(l_ncidp, "bathymetry", &l_varBId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "height", &l_varHId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "momentum_x", &l_varHuId), __FILE__, __LINE__);
    ncCheck(nc_inq_varid(l_ncidp, "momentum_y", &l_varHvId), __FILE__, __LINE__);

    // read dimensionless variables
    int l_useFWaveInt, l_useCudaInt, l_boundaryLInt, l_boundaryRInt, l_boundaryBInt, l_boundaryTInt;
    ncCheck(nc_get_var_int(l_ncidp, l_varUseFWaveId, &l_useFWaveInt), __FILE__, __LINE__);
    *o_useFWave = l_useFWaveInt != 0;
    ncCheck(nc_get_var_int(l_ncidp, l_varUseCudaId, &l_useCudaInt), __FILE__, __LINE__);
    *o_useCuda = l_useCudaInt != 0;
    ncCheck(nc_get_var_int(l_ncidp, l_varBoundaryLId, &l_boundaryLInt), __FILE__, __LINE__);
    *o_boundaryL = intToTBoundary(l_boundaryLInt);
    ncCheck(nc_get_var_int(l_ncidp, l_varBoundaryRId, &l_boundaryRInt), __FILE__, __LINE__);
    *o_boundaryR = intToTBoundary(l_boundaryRInt);
    ncCheck(nc_get_var_int(l_ncidp, l_varBoundaryBId, &l_boundaryBInt), __FILE__, __LINE__);
    *o_boundaryB = intToTBoundary(l_boundaryBInt);
    ncCheck(nc_get_var_int(l_ncidp, l_varBoundaryTId, &l_boundaryTInt), __FILE__, __LINE__);
    *o_boundaryT = intToTBoundary(l_boundaryTInt);
    ncCheck(nc_get_var_float(l_ncidp, l_varEndTimeId, o_endTime), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varWidthId, o_width), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varXOffsetId, o_xOffset), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varYOffsetId, o_yOffset), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varHMaxId, o_hMax), __FILE__, __LINE__);
    char *l_stationFilePath = new char[l_textLength]();
    ncCheck(nc_get_var_text(l_ncidp, l_varStationFilePathId, l_stationFilePath), __FILE__, __LINE__);
    *o_stationFilePath = std::string(l_stationFilePath);
    delete[] l_stationFilePath;
    int l_nFrames, l_k, l_timeStep, l_nOut, l_nFreqStation;
    ncCheck(nc_get_var_int(l_ncidp, l_varNFramesId, &l_nFrames), __FILE__, __LINE__);
    *o_nFrames = l_nFrames;
    ncCheck(nc_get_var_int(l_ncidp, l_varKId, &l_k), __FILE__, __LINE__);
    *o_k = l_k;
    ncCheck(nc_get_var_int(l_ncidp, l_varTimeStepId, &l_timeStep), __FILE__, __LINE__);
    *o_timeStep = l_timeStep;
    ncCheck(nc_get_var_int(l_ncidp, l_varNOutId, &l_nOut), __FILE__, __LINE__);
    *o_nOut = l_nOut;
    ncCheck(nc_get_var_int(l_ncidp, l_varNFreqStationId, &l_nFreqStation), __FILE__, __LINE__);
    *o_nFreqStation = l_nFreqStation;
    ncCheck(nc_get_var_float(l_ncidp, l_varSimTimeId, o_simTime), __FILE__, __LINE__);
    ncCheck(nc_get_var_int(l_ncidp, l_varMaxHoursId, o_maxHours), __FILE__, __LINE__);

    // read arrays
    *o_b = new t_real[(*o_nx) * (*o_ny)];
    *o_h = new t_real[(*o_nx) * (*o_ny)];
    *o_hu = new t_real[(*o_nx) * (*o_ny)];
    *o_hv = new t_real[(*o_nx) * (*o_ny)];

    ncCheck(nc_get_var_float(l_ncidp, l_varBId, *o_b), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varHId, *o_h), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varHuId, *o_hu), __FILE__, __LINE__);
    ncCheck(nc_get_var_float(l_ncidp, l_varHvId, *o_hv), __FILE__, __LINE__);
}

int tsunami_lab::io::NetCdf::tBoundaryToInt(tsunami_lab::t_boundary i_boundary)
{
    switch (i_boundary)
    {
    case tsunami_lab::t_boundary::OPEN:
        return 0;
    case tsunami_lab::t_boundary::WALL:
        return 1;
    default:
        return -1;
    }
}

tsunami_lab::t_boundary tsunami_lab::io::NetCdf::intToTBoundary(int i_boundary)
{
    switch (i_boundary)
    {
    case 0:
        return tsunami_lab::t_boundary::OPEN;
    case 1:
        return tsunami_lab::t_boundary::WALL;
    default:
        return tsunami_lab::t_boundary::OPEN;
    }
}

void tsunami_lab::io::NetCdf::writeCheckpoint(t_idx i_nx,
                                              t_idx i_ny,
                                              t_idx i_stride,
                                              t_idx i_ghostCellsX,
                                              t_idx i_ghostCellsY,
                                              bool i_useFWave,
                                              bool i_useCuda,
                                              tsunami_lab::t_boundary i_boundaryL,
                                              tsunami_lab::t_boundary i_boundaryR,
                                              tsunami_lab::t_boundary i_boundaryB,
                                              tsunami_lab::t_boundary i_boundaryT,
                                              tsunami_lab::t_real i_endTime,
                                              tsunami_lab::t_real i_width,
                                              tsunami_lab::t_real i_xOffset,
                                              tsunami_lab::t_real i_yOffset,
                                              tsunami_lab::t_real i_hMax,
                                              std::string i_stationFilePath,
                                              tsunami_lab::t_idx i_nFrames,
                                              tsunami_lab::t_idx i_k,
                                              tsunami_lab::t_idx i_timeStep,
                                              tsunami_lab::t_idx i_nOut,
                                              tsunami_lab::t_idx i_nFreqStation,
                                              tsunami_lab::t_real i_simTime,
                                              int i_maxHours,
                                              const t_real *i_b,
                                              const t_real *i_h,
                                              const t_real *i_hu,
                                              const t_real *i_hv)
{
    // netCdf file name: checkpoint_<iso-date>.nc
    time_t now;
    time(&now);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
    std::string l_fileName = "checkpoints/checkpoint_" + std::string(buf) + ".nc";

    // create netCdf file
    int l_ncidp = -1;
    if (!std::filesystem::exists("checkpoints"))
    {
        std::filesystem::create_directory("checkpoints");
    }
    ncCheck(nc_create(l_fileName.data(), NC_CLOBBER | NC_NETCDF4, &l_ncidp), __FILE__, __LINE__);

    // define dimensions
    int l_dimXId, l_dimYId, l_dimTextId;
    ncCheck(nc_def_dim(l_ncidp, "x", i_nx, &l_dimXId), __FILE__, __LINE__);
    ncCheck(nc_def_dim(l_ncidp, "y", i_ny, &l_dimYId), __FILE__, __LINE__);
    // +1 for the null-terminator
    ncCheck(nc_def_dim(l_ncidp, "text", i_stationFilePath.length() + 1, &l_dimTextId), __FILE__, __LINE__);

    // define dimensionless variables
    int l_varUseFWaveId, l_varUseCudaId, l_varBoundaryLId, l_varBoundaryRId, l_varBoundaryBId, l_varBoundaryTId;
    int l_varEndTimeId, l_varWidthId, l_varXOffsetId, l_varYOffsetId, l_varHMaxId;
    int l_varStationFilePathId, l_varNFramesId, l_varKId, l_varTimeStepId, l_varNOutId, l_varNFreqStationId;
    int l_varSimTimeId, l_varMaxHoursId;

    ncCheck(nc_def_var(l_ncidp, "useFWave", NC_INT, 0, nullptr, &l_varUseFWaveId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "useCuda", NC_INT, 0, nullptr, &l_varUseCudaId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "boundaryL", NC_INT, 0, nullptr, &l_varBoundaryLId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "boundaryR", NC_INT, 0, nullptr, &l_varBoundaryRId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "boundaryB", NC_INT, 0, nullptr, &l_varBoundaryBId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "boundaryT", NC_INT, 0, nullptr, &l_varBoundaryTId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "endTime", NC_FLOAT, 0, nullptr, &l_varEndTimeId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "width", NC_FLOAT, 0, nullptr, &l_varWidthId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "xOffset", NC_FLOAT, 0, nullptr, &l_varXOffsetId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "yOffset", NC_FLOAT, 0, nullptr, &l_varYOffsetId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "hMax", NC_FLOAT, 0, nullptr, &l_varHMaxId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "stationFilePath", NC_CHAR, 1, &l_dimTextId, &l_varStationFilePathId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "nFrames", NC_INT, 0, nullptr, &l_varNFramesId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "k", NC_INT, 0, nullptr, &l_varKId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "timeStep", NC_INT, 0, nullptr, &l_varTimeStepId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "nOut", NC_INT, 0, nullptr, &l_varNOutId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "nFreqStation", NC_INT, 0, nullptr, &l_varNFreqStationId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "simTime", NC_FLOAT, 0, nullptr, &l_varSimTimeId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "maxHours", NC_INT, 0, nullptr, &l_varMaxHoursId), __FILE__, __LINE__);

    int l_dim[2] = {l_dimYId, l_dimXId};

    // define arrays
    int l_varBId, l_varHId, l_varHuId, l_varHvId;
    ncCheck(nc_def_var(l_ncidp, "bathymetry", NC_FLOAT, 2, l_dim, &l_varBId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "height", NC_FLOAT, 2, l_dim, &l_varHId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "momentum_x", NC_FLOAT, 2, l_dim, &l_varHuId), __FILE__, __LINE__);
    ncCheck(nc_def_var(l_ncidp, "momentum_y", NC_FLOAT, 2, l_dim, &l_varHvId), __FILE__, __LINE__);

    ncCheck(nc_enddef(l_ncidp), __FILE__, __LINE__);

    // write dimensionless variables
    int l_useFWaveInt = i_useFWave ? 1 : 0;
    ncCheck(nc_put_var_int(l_ncidp, l_varUseFWaveId, &l_useFWaveInt), __FILE__, __LINE__);
    int l_useCudaInt = i_useCuda ? 1 : 0;
    ncCheck(nc_put_var_int(l_ncidp, l_varUseCudaId, &l_useCudaInt), __FILE__, __LINE__);
    int l_boundaryLInt = tBoundaryToInt(i_boundaryL);
    ncCheck(nc_put_var_int(l_ncidp, l_varBoundaryLId, &l_boundaryLInt), __FILE__, __LINE__);
    int l_boundaryRInt = tBoundaryToInt(i_boundaryR);
    ncCheck(nc_put_var_int(l_ncidp, l_varBoundaryRId, &l_boundaryRInt), __FILE__, __LINE__);
    int l_boundaryBInt = tBoundaryToInt(i_boundaryB);
    ncCheck(nc_put_var_int(l_ncidp, l_varBoundaryBId, &l_boundaryBInt), __FILE__, __LINE__);
    int l_boundaryTInt = tBoundaryToInt(i_boundaryT);
    ncCheck(nc_put_var_int(l_ncidp, l_varBoundaryTId, &l_boundaryTInt), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(l_ncidp, l_varEndTimeId, &i_endTime), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(l_ncidp, l_varWidthId, &i_width), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(l_ncidp, l_varXOffsetId, &i_xOffset), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(l_ncidp, l_varYOffsetId, &i_yOffset), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(l_ncidp, l_varHMaxId, &i_hMax), __FILE__, __LINE__);
    ncCheck(nc_put_var_text(l_ncidp, l_varStationFilePathId, i_stationFilePath.data()), __FILE__, __LINE__);
    int l_nFrames = i_nFrames;
    ncCheck(nc_put_var_int(l_ncidp, l_varNFramesId, &l_nFrames), __FILE__, __LINE__);
    int l_k = i_k;
    ncCheck(nc_put_var_int(l_ncidp, l_varKId, &l_k), __FILE__, __LINE__);
    int l_timeStep = i_timeStep;
    ncCheck(nc_put_var_int(l_ncidp, l_varTimeStepId, &l_timeStep), __FILE__, __LINE__);
    int l_nOut = i_nOut;
    ncCheck(nc_put_var_int(l_ncidp, l_varNOutId, &l_nOut), __FILE__, __LINE__);
    int l_nFreqStation = i_nFreqStation;
    ncCheck(nc_put_var_int(l_ncidp, l_varNFreqStationId, &l_nFreqStation), __FILE__, __LINE__);
    ncCheck(nc_put_var_float(l_ncidp, l_varSimTimeId, &i_simTime), __FILE__, __LINE__);
    ncCheck(nc_put_var_int(l_ncidp, l_varMaxHoursId, &i_maxHours), __FILE__, __LINE__);

    // write arrays
    t_idx start_p[2] = {0, 0};
    t_idx count_p[2] = {1, i_nx};
    for (start_p[0] = 0; start_p[0] < i_ny; ++start_p[0])
    {
        ncCheck(nc_put_vara_float(l_ncidp, l_varBId, start_p, count_p, i_b + (start_p[0] + i_ghostCellsY) * i_stride + i_ghostCellsX), __FILE__, __LINE__);
        ncCheck(nc_put_vara_float(l_ncidp, l_varHId, start_p, count_p, i_h + (start_p[0] + i_ghostCellsY) * i_stride + i_ghostCellsX), __FILE__, __LINE__);
        ncCheck(nc_put_vara_float(l_ncidp, l_varHuId, start_p, count_p, i_hu + (start_p[0] + i_ghostCellsY) * i_stride + i_ghostCellsX), __FILE__, __LINE__);
        ncCheck(nc_put_vara_float(l_ncidp, l_varHvId, start_p, count_p, i_hv + (start_p[0] + i_ghostCellsY) * i_stride + i_ghostCellsX), __FILE__, __LINE__);
    }
    ncCheck(nc_close(l_ncidp), __FILE__, __LINE__);
}