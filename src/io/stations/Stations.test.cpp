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
#include "../csv/Csv.h"

TEST_CASE("Test Reading the test.json", "[StationsRead]")
{
    std::string l_path = "src/data/test.json";
    tsunami_lab::io::Stations l_stations(l_path);

    REQUIRE(l_stations.getT() == 2);
    REQUIRE(l_stations.getStations().size() == 2);
    REQUIRE(l_stations.getStations()[0].name == "Test_1");
    REQUIRE(l_stations.getStations()[0].x == 0);
    REQUIRE(l_stations.getStations()[0].y == 2);
    REQUIRE(l_stations.getStations()[1].name == "Test_2");
    REQUIRE(l_stations.getStations()[1].x == 5);
    REQUIRE(l_stations.getStations()[1].y == 0);
}

TEST_CASE("Test Writing JSons for all Stations", "[StationsWrite]")
{
    std::string l_path = "src/data/test.json";
    tsunami_lab::io::Stations l_stations(l_path);

    tsunami_lab::t_real l_h[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tsunami_lab::t_real l_hu[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tsunami_lab::t_real l_hv[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tsunami_lab::t_real l_b[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    l_stations.init(1, 3, 3, 3, 0, 0, 0, 0, l_b, false);

    l_stations.write(1, l_h, l_hu, l_hv);
    l_stations.write(2, l_h, l_hu, l_hv);
    l_stations.write(3, l_h, l_hu, l_hv);

    REQUIRE(std::filesystem::exists("stations/station_Test_1.csv"));
    REQUIRE(std::filesystem::exists("stations/station_Test_2.csv"));

    rapidcsv::Document test1Doc, test2Doc;

    tsunami_lab::io::Csv::openCSV("stations/station_Test_1.csv", test1Doc, true);
    tsunami_lab::io::Csv::openCSV("stations/station_Test_2.csv", test2Doc, true);

    REQUIRE(test1Doc.GetRowCount() == 3);
    REQUIRE(test2Doc.GetRowCount() == 0);
    tsunami_lab::t_real l_height, l_momentum_x, l_momentum_y, l_bathymetry, l_time;
    l_time = test1Doc.GetCell<tsunami_lab::t_real>(0, 0);
    l_height = test1Doc.GetCell<tsunami_lab::t_real>(1, 0);
    l_momentum_x = test1Doc.GetCell<tsunami_lab::t_real>(2, 0);
    l_momentum_y = test1Doc.GetCell<tsunami_lab::t_real>(3, 0);
    l_bathymetry = test1Doc.GetCell<tsunami_lab::t_real>(4, 0);
    REQUIRE(l_time == 1);
    REQUIRE(l_height == 7);
    REQUIRE(l_momentum_x == 7);
    REQUIRE(l_momentum_y == 7);
    REQUIRE(l_bathymetry == 7);
}