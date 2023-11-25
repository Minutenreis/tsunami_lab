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

void tsunami_lab::io::Csv::init(t_real i_dxy,
                                t_idx i_nx,
                                t_idx i_ny,
                                t_idx i_stride,
                                t_idx i_ghostCellsX,
                                t_idx i_ghostCellsY,
                                t_real i_offsetX,
                                t_real i_offsetY,
                                t_real const *i_b)
{
  // save setup parameters
  m_dxy = i_dxy;
  m_nx = i_nx;
  m_ny = i_ny;
  m_stride = i_stride;
  m_ghostCellsX = i_ghostCellsX;
  m_ghostCellsY = i_ghostCellsY;
  m_offsetX = i_offsetX;
  m_offsetY = i_offsetY;
  m_b = i_b;
}

void tsunami_lab::io::Csv::write(t_real const *i_h,
                                 t_real const *i_hu,
                                 t_real const *i_hv,
                                 t_real,
                                 t_idx i_nOut)
{
  std::string l_path = "solutions/solution_" + std::to_string(i_nOut) + ".csv";
  std::cout << "  writing wave field to " << l_path << std::endl;

  std::ofstream io_stream;
  io_stream.open(l_path);

  // write the CSV header
  io_stream << "x,y";
  if (i_h != nullptr)
    io_stream << ",height";
  if (i_hu != nullptr)
    io_stream << ",momentum_x";
  if (i_hv != nullptr)
    io_stream << ",momentum_y";
  if (m_b != nullptr)
    io_stream << ",bathymetry";
  io_stream << "\n";

  // iterate over all cells
  for (t_idx l_iy = m_ghostCellsY; l_iy < m_ny + m_ghostCellsY; l_iy++)
  {
    for (t_idx l_ix = m_ghostCellsX; l_ix < m_nx + m_ghostCellsX; l_ix++)
    {
      // derive coordinates of cell center
      t_real l_posX = (l_ix - m_ghostCellsX + 0.5) * m_dxy + m_offsetX; // ghost cells don't count for distance
      t_real l_posY = (l_iy - m_ghostCellsY + 0.5) * m_dxy + m_offsetY; // ghost cells don't count for distance

      t_idx l_id = l_ix + l_iy * m_stride;

      // write data
      io_stream << l_posX << "," << l_posY;
      if (i_h != nullptr)
        io_stream << "," << i_h[l_id];
      if (i_hu != nullptr)
        io_stream << "," << i_hu[l_id];
      if (i_hv != nullptr)
        io_stream << "," << i_hv[l_id];
      if (m_b != nullptr)
        io_stream << "," << m_b[l_id];
      io_stream << "\n";
    }
  }
  io_stream << std::flush;
  io_stream.close();
}

void tsunami_lab::io::Csv::openCSV(const std::string &i_filePath, rapidcsv::Document &o_doc, bool header)
{
  if (header)
  {
    o_doc = rapidcsv::Document(i_filePath, rapidcsv::LabelParams(0, -1));
  }
  else
  {
    o_doc = rapidcsv::Document(i_filePath, rapidcsv::LabelParams(-1, -1));
  }
}