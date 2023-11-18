/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Stations Controller Class tests
 */
#include <catch2/catch.hpp>
#include "Stations.h"

TEST_CASE("Test Reading the Json", "[StationsRead]")
{
    std::string l_path = "src/data/test.json";
    tsunami_lab::io::Stations l_stations(l_path);

    REQUIRE(l_stations.getT() == 2);
    REQUIRE(l_stations.getStations().size() == 2);
    REQUIRE(l_stations.getStations()[0].name == "Radio 1");
    REQUIRE(l_stations.getStations()[0].x == 9);
    REQUIRE(l_stations.getStations()[0].y == 0);
    REQUIRE(l_stations.getStations()[1].name == "Radio 2");
    REQUIRE(l_stations.getStations()[1].x == 14);
    REQUIRE(l_stations.getStations()[1].y == 0);
}