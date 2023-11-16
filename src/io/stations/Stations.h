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
#include "json.hpp"

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
    t_real m_T = 0.0;

    //! station data vector
    std::vector<t_station> m_stations;

public:
    /**
     * Constructs the stations controller.
     *
     * @param path path to the stations file
     */
    Stations(const std::string path);

    /**
     * Destructor
     */
    ~Stations();
};

#endif
