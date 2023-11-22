/**
 * @author Alexander Breuer (alex.breuer AT uni-jena.de)
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Unit tests for the CSV-interface.
 **/
#include <catch2/catch.hpp>
#include "../../constants.h"
#include <sstream>
#include <filesystem>

#define private public
#include "Csv.h"
#undef public

TEST_CASE("Test the CSV-writer for 1D settings.", "[CsvWrite1d]")
{
  // define a simple example
  tsunami_lab::t_real l_h[7] = {0, 1, 2, 3, 4, 5, 6};
  tsunami_lab::t_real l_hu[7] = {6, 5, 4, 3, 2, 1, 0};

  // delete old stations
  if (std::filesystem::exists("solutions"))
  {
    std::filesystem::remove_all("solutions");
  }
  std::filesystem::create_directory("solutions");

  tsunami_lab::io::Csv *l_csv = new tsunami_lab::io::Csv();

  l_csv->init(0.5,
              5,
              1,
              7,
              1,
              0,
              0,
              0,
              nullptr);

  std::stringstream l_stream0;
  l_csv->write(l_h,
               l_hu,
               nullptr,
               0,
               0);

  REQUIRE(std::filesystem::exists("solutions/solution_0.csv"));
  rapidcsv::Document testDoc;
  tsunami_lab::io::Csv::openCSV("solutions/solution_0.csv", testDoc, true);

  REQUIRE(testDoc.GetRowCount() == 5);
  REQUIRE(testDoc.GetColumnCount() == 4);

  for (tsunami_lab::t_idx l_row = 0; l_row < 5; l_row++)
  {
    REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(0, l_row) == (l_row + 0.5) * 0.5);
    REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(1, l_row) == 0.25);
    REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(2, l_row) == l_h[l_row + 1]);
    REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(3, l_row) == l_hu[l_row + 1]);
  }

  delete l_csv;
}

TEST_CASE("Test the CSV-writer for 2D settings.", "[CsvWrite2d]")
{
  // define a simple example
  tsunami_lab::t_real l_h[16] = {0, 1, 2, 3,
                                 4, 5, 6, 7,
                                 8, 9, 10, 11,
                                 12, 13, 14, 15};
  tsunami_lab::t_real l_hu[16] = {15, 14, 13, 12,
                                  11, 10, 9, 8,
                                  7, 6, 5, 4,
                                  3, 2, 1, 0};
  tsunami_lab::t_real l_hv[16] = {0, 4, 8, 12,
                                  1, 5, 9, 13,
                                  2, 6, 10, 14,
                                  3, 7, 11, 15};

  // delete old stations
  if (std::filesystem::exists("solutions"))
  {
    std::filesystem::remove_all("solutions");
  }
  std::filesystem::create_directory("solutions");

  tsunami_lab::io::Csv *l_csv = new tsunami_lab::io::Csv();

  l_csv->init(10,
              2,
              2,
              4,
              1,
              1,
              0,
              0,
              nullptr);

  l_csv->write(l_h,
               l_hu,
               l_hv,
               0,
               0);

  REQUIRE(std::filesystem::exists("solutions/solution_0.csv"));

  rapidcsv::Document testDoc;
  tsunami_lab::io::Csv::openCSV("solutions/solution_0.csv", testDoc, true);

  REQUIRE(testDoc.GetRowCount() == 4);
  REQUIRE(testDoc.GetColumnCount() == 5);

  tsunami_lab::t_idx l_row = 0;

  for (tsunami_lab::t_idx l_y = 0; l_y < 2; l_y++)
    for (tsunami_lab::t_idx l_x = 0; l_x < 2; l_x++)
    {
      REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(0, l_row) == (l_x + 0.5) * 10);
      REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(1, l_row) == (l_y + 0.5) * 10);
      REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(2, l_row) == l_h[(l_x + 1) + (l_y + 1) * 4]);
      REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(3, l_row) == l_hu[(l_x + 1) + (l_y + 1) * 4]);
      REQUIRE(testDoc.GetCell<tsunami_lab::t_real>(4, l_row) == l_hv[(l_x + 1) + (l_y + 1) * 4]);
      l_row++;
    }
}

TEST_CASE("Test the CSV-reader with a 4 column file.", "[CsvRead4Columns]")
{
  rapidcsv::Document doc;
  tsunami_lab::io::Csv::openCSV("src/data/test.csv", doc, false);
  tsunami_lab::t_real i_row = 2;
  tsunami_lab::t_real bathometry = doc.GetCell<tsunami_lab::t_real>(3, i_row);
  REQUIRE(bathometry == -5.84086714415f);
}
