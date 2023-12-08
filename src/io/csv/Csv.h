/**
 * @author Alexander Breuer (alex.breuer AT uni-jena.de)
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * IO-routines for writing a snapshot as Comma Separated Values (CSV).
 **/
#ifndef TSUNAMI_LAB_IO_CSV
#define TSUNAMI_LAB_IO_CSV

#include "../../constants.h"
#include "../IoWriter.h"
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include "rapidcsv.h"

namespace tsunami_lab
{
  namespace io
  {
    class Csv;
  }
}

class tsunami_lab::io::Csv : public IoWriter
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

  //! file stream
  char *m_outputPath;

  //! bathymetry.
  t_real const *m_b;

public:
  /**
   * @brief Initialize the CSV File.
   *
   * @param i_dxy cell size.
   * @param i_nx number of cells in x-direction.
   * @param i_ny number of cells in y-direction.
   * @param i_stride stride of the data.
   * @param i_ghostCellsX number of ghost cells in x-direction.
   * @param i_ghostCellsY number of ghost cells in y-direction.
   * @param i_offsetX offset in x-direction.
   * @param i_offsetY offset in y-direction.
   * @param i_b bathymetry.
   * @param i_useCheckpoint flag if checkpoint is used.
   */
  void init(t_real i_dxy,
            t_idx i_nx,
            t_idx i_ny,
            t_idx i_stride,
            t_idx i_ghostCellsX,
            t_idx i_ghostCellsY,
            t_real i_offsetX,
            t_real i_offsetY,
            t_real,
            t_real const *i_b,
            bool i_useCheckpoint);

  /**
   * @brief Writes the data to the output.
   *
   * @param i_h water height.
   * @param i_hu momentum in x-direction.
   * @param i_hv momentum in y-direction.
   * @param i_nOut number of the output.
   */
  void write(t_real const *i_h,
             t_real const *i_hu,
             t_real const *i_hv,
             t_real,
             t_idx i_nOut);

  /**
   * @brief gets rapidcsv::Document and row count from CSV file
   *
   * @param i_filePath path to CSV file
   * @param o_doc csv file as rapidcsv::Document
   */
  static void openCSV(const std::string &i_filePath, rapidcsv::Document &o_doc, bool header);
};

#endif