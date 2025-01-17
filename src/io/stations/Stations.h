/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Stations Controller Class to get data at specific points in space
 */
#ifndef TSUNAMI_LAB_IO_STATIONS
#define TSUNAMI_LAB_IO_STATIONS

#include <string>
#include <vector>
#include "../../constants.h"
#include <nlohmann/json.hpp>

namespace tsunami_lab
{
    namespace io
    {
        class Stations;
    }

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(t_station, name, x, y)
}

class tsunami_lab::io::Stations
{
private:
    //! output period
    t_real m_T = 0;

    //! station data vector
    std::vector<t_station> m_stations;

    //! has stations
    bool m_hasStations = false;

    //! stations file path
    const std::string m_path;

    //! cell width in x- and y-direction.
    const t_real m_dxy;

    //! number of cells in x-direction.
    const t_idx m_nx;

    //! number of cells in y-direction.
    const t_idx m_ny;

    //! stride of the data arrays.
    const t_idx m_stride;

    //! number of ghost cells in x-direction.
    const t_idx m_ghostCellsX;

    //! number of ghost cells in y-direction.
    const t_idx m_ghostCellsY;

    //! offset in x-direction.
    const t_real m_offsetX;

    //! offset in y-direction.
    const t_real m_offsetY;

    //! bathymetry.
    const t_real *m_b;

public:
    /**
     * Constructs the stations controller.
     *
     * @param path path to the stations file
     * @param i_dxy cell width in x- and y-direction.
     * @param i_nx number of cells in x-direction.
     * @param i_ny number of cells in y-direction.
     * @param i_stride stride of the data arrays in y-direction (x is assumed to be stride-1).
     * @param i_ghostCellsX number of ghost cells in x-direction.
     * @param i_ghostCellsY number of ghost cells in y-direction.
     * @param i_offsetX offset in x-direction.
     * @param i_offsetY offset in y-direction.
     * @param i_b bathymetry of the cells; optional: use nullptr if not required.
     * @param i_useCheckpoint flag if checkpoint is used.
     */
    Stations(const std::string path,
             t_real i_dxy,
             t_idx i_nx,
             t_idx i_ny,
             t_idx i_stride,
             t_idx i_ghostCellsX,
             t_idx i_ghostCellsY,
             t_real i_offsetX,
             t_real i_offsetY,
             t_real const *i_b,
             bool i_useCheckpoint);

    /**
     * Returns the output period.
     *
     * @return output period
     */
    t_real getT() const;

    /**
     * Returns the station data vector.
     *
     * @return station data vector
     */
    std::vector<t_station> getStations() const;

    /**
     * @brief Returns the number of stations.
     */
    bool hasStations() const;

    /**
     * @brief Returns the path to the stations file.
     */
    std::string getPath() const;

    /**
     * Writes the data as CSV to the given stream.
     *
     * @param i_simTime simulation time.
     * @param i_h water height of the cells; optional: use nullptr if not required.
     * @param i_hu momentum in x-direction of the cells; optional: use nullptr if not required.
     * @param i_hv momentum in y-direction of the cells; optional: use nullptr if not required.
     **/
    void write(t_real i_simTime,
               t_real const *i_h,
               t_real const *i_hu,
               t_real const *i_hv);

    /**
     * @brief Initializes output files
     *

     */
    void init();
};

#endif
