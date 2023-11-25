/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Stations Controller Class to get data at specific points in space
 */
#include "Stations.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

using json = nlohmann::json;

tsunami_lab::io::Stations::Stations(const std::string path)
{
    std::ifstream f(path);
    json data = json::parse(f);

    m_T = data["period"];

    m_stations = data["stations"];
}

tsunami_lab::t_real tsunami_lab::io::Stations::getT() const
{
    return m_T;
}

std::vector<tsunami_lab::t_station> tsunami_lab::io::Stations::getStations() const
{
    return m_stations;
}

void tsunami_lab::io::Stations::init()
{
    for (t_station l_station : m_stations)
    {
        std::string l_path = "stations/station_" + l_station.name + ".csv";
        std::ofstream l_file;
        l_file.open(l_path);
        l_file << "time,height,momentum_x,momentum_y,bathymetry" << std::endl;
    }
}

void tsunami_lab::io::Stations::write(t_real i_dxy,
                                      t_idx i_nx,
                                      t_idx i_ny,
                                      t_idx i_stride,
                                      t_idx i_ghostCellsX,
                                      t_idx i_ghostCellsY,
                                      t_real i_simTime,
                                      t_real i_offsetX,
                                      t_real i_offsetY,
                                      t_real const *i_h,
                                      t_real const *i_hu,
                                      t_real const *i_hv,
                                      t_real const *i_b)
{
    for (t_station l_station : m_stations)
    {
        if (l_station.x - i_offsetX < 0 || l_station.x - i_offsetX >= i_nx * i_dxy || l_station.y - i_offsetY < 0 || l_station.y - i_offsetY >= i_ny * i_dxy)
            continue; // station is outside of the domain

        t_idx l_ix = (l_station.x - i_offsetX) / i_dxy + i_ghostCellsX;
        t_idx l_iy = (l_station.y - i_offsetY) / i_dxy + i_ghostCellsY;

        t_idx l_id = l_ix + l_iy * i_stride;

        std::string l_path = "stations/station_" + l_station.name + ".csv";
        std::ofstream l_file;
        l_file.open(l_path, std::ios_base::app);
        l_file << i_simTime;

        if (i_h != nullptr)
            l_file << "," << i_h[l_id];
        else
            l_file << ",0";
        if (i_hu != nullptr)
            l_file << "," << i_hu[l_id];
        else
            l_file << ",0";
        if (i_hv != nullptr)
            l_file << "," << i_hv[l_id];
        else
            l_file << ",0";
        if (i_b != nullptr)
            l_file << "," << i_b[l_id];
        else
            l_file << ",0";
        l_file << std::endl
               << std::flush;
    }
}