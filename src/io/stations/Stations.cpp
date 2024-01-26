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

tsunami_lab::io::Stations::Stations(const std::string path,
                                    t_real i_dxy,
                                    t_idx i_nx,
                                    t_idx i_ny,
                                    t_idx i_stride,
                                    t_idx i_ghostCellsX,
                                    t_idx i_ghostCellsY,
                                    t_real i_offsetX,
                                    t_real i_offsetY,
                                    t_real const *i_b,
                                    bool i_useCheckpoint) : m_path(path),
                                                            m_dxy(i_dxy),
                                                            m_nx(i_nx),
                                                            m_ny(i_ny),
                                                            m_stride(i_stride),
                                                            m_ghostCellsX(i_ghostCellsX),
                                                            m_ghostCellsY(i_ghostCellsY),
                                                            m_offsetX(i_offsetX),
                                                            m_offsetY(i_offsetY),
                                                            m_b(i_b)
{
    m_path = path;
    std::ifstream f(path);
    json data = json::parse(f);

    m_T = data["period"];

    m_stations = data["stations"];
    // remove stations outside of the domain
    m_stations.erase(std::remove_if(m_stations.begin(), m_stations.end(), [&](t_station i_station)
                                    { return i_station.x - m_offsetX < 0 || i_station.x - m_offsetX >= m_nx * m_dxy || i_station.y - m_offsetY < 0 || i_station.y - m_offsetY >= m_ny * m_dxy; }),
                     m_stations.end());
    m_hasStations = m_stations.size() > 0;

    // if checkpoint is used, the stations are already initialized
    if (!i_useCheckpoint && m_hasStations)
    {
        // delete old stations
        if (std::filesystem::exists("stations"))
        {
            std::filesystem::remove_all("stations");
        }
        std::filesystem::create_directory("stations");
        for (t_station l_station : m_stations)
        {
            std::string l_path = "stations/station_" + l_station.name + ".csv";
            std::ofstream l_file;
            l_file.open(l_path);
            l_file << "time,height,momentum_x,momentum_y,bathymetry" << std::endl;
        }
    }
}

tsunami_lab::t_real tsunami_lab::io::Stations::getT() const
{
    return m_T;
}

std::vector<tsunami_lab::t_station> tsunami_lab::io::Stations::getStations() const
{
    return m_stations;
}

bool tsunami_lab::io::Stations::hasStations() const
{
    return m_hasStations;
}

std::string tsunami_lab::io::Stations::getPath() const
{
    return m_path;
}

void tsunami_lab::io::Stations::write(t_real i_simTime,
                                      t_real const *i_h,
                                      t_real const *i_hu,
                                      t_real const *i_hv)
{
    for (t_station l_station : m_stations)
    {
        t_idx l_ix = (l_station.x - m_offsetX) / m_dxy + m_ghostCellsX;
        t_idx l_iy = (l_station.y - m_offsetY) / m_dxy + m_ghostCellsY;

        t_idx l_id = l_ix + l_iy * m_stride;

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
        if (m_b != nullptr)
            l_file << "," << m_b[l_id];
        else
            l_file << ",0";
        l_file << std::endl
               << std::flush;
    }
}
