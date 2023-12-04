/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * IO-routines for reading / writing netCdf data.
 **/

#ifndef TSUNAMI_LAB_IO_NETCDF
#define TSUNAMI_LAB_IO_NETCDF

#include "../../constants.h"
#include "../IoWriter.h"

namespace tsunami_lab
{
    namespace io
    {
        class NetCdf;
    }
}

class tsunami_lab::io::NetCdf : public IoWriter
{
private:
    //! cell width in x- and y-direction.
    t_real m_dxy;

    //! number of cells in x-direction.
    t_idx m_nx;

    //! number of cells in y-direction.
    t_idx m_ny;

    //! stride of the data arrays.
    t_idx m_stride;

    //! number of ghost cells in x-direction.
    t_idx m_ghostCellsX;

    //! number of ghost cells in y-direction.
    t_idx m_ghostCellsY;

    //! offset in x-direction.
    t_real m_offsetX;

    //! offset in y-direction.
    t_real m_offsetY;

    //! cell size to be averaged.
    t_idx m_k;

    //! netCdf file id.
    int m_ncidp = -1;

    //! netCdf dimension ids.
    int m_varXId, m_varYId, m_varTimeId, m_varHId, m_varHuId, m_varHvId, m_varBId;

    /**
     * @brief Prune Ghost Cells of Data
     *
     * @param i_data input data
     * @param i_var variable id to input
     * @param i_nOut output time step
     * @param i_hasTime true if data has time
     */
    void putVaraWithGhostcells(t_real const *i_data, int i_var, t_idx i_nOut, bool i_hasTime);

public:
    /**
     * @brief Initialize the netCdf File.
     *
     * @param i_dxy cell size.
     * @param i_nx number of cells in x-direction.
     * @param i_ny number of cells in y-direction.
     * @param i_stride stride of the data.
     * @param i_ghostCellsX number of ghost cells in x-direction.
     * @param i_ghostCellsY number of ghost cells in y-direction.
     * @param i_offsetX offset in x-direction.
     * @param i_offsetY offset in y-direction.
     * @param i_k cell size to be averaged.
     * @param i_b bathymetry.
     */
    void init(t_real i_dxy,
              t_idx i_nx,
              t_idx i_ny,
              t_idx i_stride,
              t_idx i_ghostCellsX,
              t_idx i_ghostCellsY,
              t_real i_offsetX,
              t_real i_offsetY,
              t_real i_k,
              t_real const *i_b);

    /**
     * @brief Writes the data to the output.
     *
     * @param i_h water height.
     * @param i_hu momentum in x-direction.
     * @param i_hv momentum in y-direction.
     * @param i_time time in simulation.
     */
    void write(t_real const *i_h,
               t_real const *i_hu,
               t_real const *i_hv,
               t_real i_time,
               t_idx i_nOut);

    /**
     * @brief read 3D data from netCdf file.
     *
     * @param i_fileName name of the netCdf file.
     * @param o_nx number of cells in x-direction.
     * @param o_ny number of cells in y-direction.
     * @param o_nz number of cells in z-direction.
     * @param o_x pointer to array of x-coordinates (Important: Gets a new dynamically allocated array written on it).
     * @param o_y pointer to array of y-coordinates (Important: Gets a new dynamically allocated array written on it).
     * @param o_z pointer to array of z-coordinates (Important: Gets a new dynamically allocated array written on it).
     */
    static void read(char *i_fileName,
                     t_idx *o_nx,
                     t_idx *o_ny,
                     t_real **o_x,
                     t_real **o_y,
                     t_real **o_z);

    /**
     * @brief checks if netCdf operation was successful and prints error on failure
     *
     * @param i_status status code of the operation
     * @param i_file file name of the operation
     * @param i_line line number of the operation
     */
    static void ncCheck(int i_status,
                        char const *i_file,
                        int i_line);

    static void readCheckpoint(char *i_fileName,
                               t_idx *o_nx,
                               t_idx *o_ny,
                               bool *o_useFWave,
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
                               t_real **o_hv);

    static void writeCheckpoint(t_idx i_nx,
                                t_idx i_ny,
                                bool i_useFWave,
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
                                t_real *i_b,
                                t_real *i_h,
                                t_real *i_hu,
                                t_real *i_hv);
};

#endif