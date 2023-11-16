/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Stations Controller Class to get data at specific points in space
 */
#include "Stations.h"
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

tsunami_lab::io::Stations::Stations(const std::string path)
{
    std::ifstream f(path);
    json data = json::parse(f);

    t_real l_frequency = data["frequency"];

    m_T = 1.0 / l_frequency;
    m_stations = data["stations"];
}

tsunami_lab::io::Stations::~Stations()
{
}