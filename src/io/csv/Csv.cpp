/**
 * @author Alexander Breuer (alex.breuer AT uni-jena.de)
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * IO-routines for reading / writing a snapshot as Comma Separated Values (CSV).
 **/
#include "Csv.h"

void tsunami_lab::io::Csv::write(t_real i_dxy,
                                 t_idx i_nx,
                                 t_idx i_ny,
                                 t_idx i_stride,
                                 t_idx i_ghostCellsX,
                                 t_idx i_ghostCellsY,
                                 t_real i_offsetX,
                                 t_real i_offsetY,
                                 t_real const *i_h,
                                 t_real const *i_hu,
                                 t_real const *i_hv,
                                 t_real const *i_b,
                                 std::ostream &io_stream)
{
  // write the CSV header
  io_stream << "x,y";
  if (i_h != nullptr)
    io_stream << ",height";
  if (i_hu != nullptr)
    io_stream << ",momentum_x";
  if (i_hv != nullptr)
    io_stream << ",momentum_y";
  if (i_b != nullptr)
    io_stream << ",bathymetry";
  io_stream << "\n";

  // iterate over all cells
  for (t_idx l_iy = i_ghostCellsY; l_iy < i_ny + i_ghostCellsY; l_iy++)
  {
    for (t_idx l_ix = i_ghostCellsX; l_ix < i_nx + i_ghostCellsX; l_ix++)
    {
      // derive coordinates of cell center
      t_real l_posX = (l_ix - i_ghostCellsX + 0.5) * i_dxy + i_offsetX; // ghost cells don't count for distance
      t_real l_posY = (l_iy - i_ghostCellsY + 0.5) * i_dxy + i_offsetY; // ghost cells don't count for distance

      t_idx l_id = l_ix + l_iy * i_stride;

      // write data
      io_stream << l_posX << "," << l_posY;
      if (i_h != nullptr)
        io_stream << "," << i_h[l_id];
      if (i_hu != nullptr)
        io_stream << "," << i_hu[l_id];
      if (i_hv != nullptr)
        io_stream << "," << i_hv[l_id];
      if (i_b != nullptr)
        io_stream << "," << i_b[l_id];
      io_stream << "\n";
    }
  }
  io_stream << std::flush;
}

void tsunami_lab::io::Csv::openCSV(const std::string &i_filePath, rapidcsv::Document &o_doc, size_t &o_rowCount, bool header)
{
  if (header)
  {
    o_doc = rapidcsv::Document(i_filePath, rapidcsv::LabelParams(0, -1));
  }
  else
  {
    o_doc = rapidcsv::Document(i_filePath, rapidcsv::LabelParams(-1, -1));
  }
  o_rowCount = o_doc.GetRowCount();
}