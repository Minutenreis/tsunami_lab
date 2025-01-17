/**
 * @author Alexander Breuer (alex.breuer AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Entry-point for simulations.
 **/
#include "patches/wavePropagation1d/WavePropagation1d.h"
#include "patches/wavePropagation2d/WavePropagation2d.h"
#include "patches/wavePropagationCUDA/WavePropagationCUDA.h"
#include "setups/damBreak1d/DamBreak1d.h"
#include "setups/rareRare1d/RareRare1d.h"
#include "setups/shockShock1d/ShockShock1d.h"
#include "setups/custom1d/Custom1d.h"
#include "setups/supercritical1d/Supercritical1d.h"
#include "setups/subcritical1d/Subcritical1d.h"
#include "setups/tsunamiEvent1d/TsunamiEvent1d.h"
#include "setups/damBreak2d/DamBreak2d.h"
#include "setups/artificialTsunami2d/ArtificialTsunami2d.h"
#include "setups/tsunamiEvent2d/TsunamiEvent2d.h"
#include "io/csv/Csv.h"
#include "io/stations/Stations.h"
#include "io/netCdf/NetCdf.h"
#include "constants.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>
#include <limits>
#include <algorithm>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <omp.h>

// converts a string to a boundary condition (tsunami_lab::t_boundary)
void getBoundary(std::string i_name, tsunami_lab::t_boundary *o_boundary)
{
  if (i_name == "WALL")
  {
    *o_boundary = tsunami_lab::t_boundary::WALL;
  }
  else if (i_name == "OPEN")
  {
    *o_boundary = tsunami_lab::t_boundary::OPEN;
  }
  else
  {
    // default is OPEN
    *o_boundary = tsunami_lab::t_boundary::OPEN;
  }
}

// prints "i_message: xh xmin xs xms"
void printTime(std::chrono::nanoseconds i_duration, std::string i_message)
{
  std::cout << i_message << ": ";
  if (i_duration > std::chrono::hours(1))
    std::cout << std::chrono::duration_cast<std::chrono::hours>(i_duration).count() << "h ";
  if (i_duration > std::chrono::minutes(1))
    std::cout << std::chrono::duration_cast<std::chrono::minutes>(i_duration).count() % 60 << "min ";
  if (i_duration > std::chrono::seconds(1))
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(i_duration).count() % 60 << "s ";
  if (i_duration > std::chrono::milliseconds(1))
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(i_duration).count() % 1000 << "ms ";
  if (i_duration > std::chrono::microseconds(1))
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(i_duration).count() % 1000 << "us ";
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(i_duration).count() % 1000 << "ns" << std::endl;
}

int main(int i_argc,
         char *i_argv[])
{
  auto l_start = std::chrono::high_resolution_clock::now();
  // number of cells in x- and y-direction

  std::cout << "####################################" << std::endl;
  std::cout << "### Tsunami Lab                  ###" << std::endl;
  std::cout << "###                              ###" << std::endl;
  std::cout << "### https://scalable.uni-jena.de ###" << std::endl;
  std::cout << "####################################" << std::endl;

  // defaults
  bool l_useFwave = true;
  tsunami_lab::t_boundary l_boundaryL = tsunami_lab::t_boundary::OPEN;
  tsunami_lab::t_boundary l_boundaryR = tsunami_lab::t_boundary::OPEN;
  tsunami_lab::t_boundary l_boundaryB = tsunami_lab::t_boundary::OPEN;
  tsunami_lab::t_boundary l_boundaryT = tsunami_lab::t_boundary::OPEN;
  tsunami_lab::t_real l_endTime = 1.25;
  tsunami_lab::t_real l_width = 10.0; // width in x direction, y is scaled by l_ny / l_nx
  tsunami_lab::t_real l_xOffset = 0;
  tsunami_lab::t_real l_yOffset = 0;
  tsunami_lab::t_real l_hMax = std::numeric_limits<tsunami_lab::t_real>::lowest();
  tsunami_lab::setups::Setup *l_setup = nullptr;
  tsunami_lab::io::Stations *l_stations = nullptr;
  tsunami_lab::io::IoWriter *l_writer = nullptr;
  tsunami_lab::patches::WavePropagation *l_waveProp = nullptr;
  tsunami_lab::t_idx l_nFrames = 100;
  tsunami_lab::t_idx l_k = 1;
  tsunami_lab::t_idx l_timeStep = 0;
  tsunami_lab::t_idx l_nOut = 0;
  tsunami_lab::t_idx l_nx = 1;
  tsunami_lab::t_idx l_ny = 1;
  std::string l_stationsPath;
  int l_maxHours = 24;
  bool l_useCheckpoint = false;
  bool l_useNetCdf = true;
  bool l_useFileIO = true;
  bool l_useCuda = false;

  std::cout << "runtime configuration" << std::endl;
  std::cout << "  Number of Threads: " << omp_get_max_threads() << std::endl;

  if (std::filesystem::exists("checkpoints") && std::filesystem::is_directory("checkpoints") && !std::filesystem::is_empty("checkpoints"))
  {
    l_useCheckpoint = true;

    // find newest checkpoint
    std::vector<std::string> l_checkpoints = {};
    for (const auto &entry : std::filesystem::directory_iterator("checkpoints"))
    {
      l_checkpoints.push_back(entry.path());
    }
    std::sort(l_checkpoints.begin(), l_checkpoints.end());
    std::string l_newestCheckpoint = l_checkpoints.back();

    // load checkpoint
    std::cout << "  loading checkpoint " << l_newestCheckpoint << std::endl;

    tsunami_lab::t_real *l_b = nullptr;
    tsunami_lab::t_real *l_h = nullptr;
    tsunami_lab::t_real *l_hu = nullptr;
    tsunami_lab::t_real *l_hv = nullptr;

    tsunami_lab::io::NetCdf::readCheckpoint(l_newestCheckpoint.data(),
                                            &l_nx,
                                            &l_ny,
                                            &l_useFwave,
                                            &l_useCuda,
                                            &l_boundaryL,
                                            &l_boundaryR,
                                            &l_boundaryB,
                                            &l_boundaryT,
                                            &l_endTime,
                                            &l_width,
                                            &l_xOffset,
                                            &l_yOffset,
                                            &l_hMax,
                                            &l_stationsPath,
                                            &l_nFrames,
                                            &l_k,
                                            &l_timeStep,
                                            &l_nOut,
                                            &l_maxHours,
                                            &l_b,
                                            &l_h,
                                            &l_hu,
                                            &l_hv);

    // always netCdf 2D output
    l_useNetCdf = true;
    if (!l_useCuda)
    {
      l_waveProp = new tsunami_lab::patches::WavePropagation2d(l_nx, l_ny, l_useFwave, l_boundaryL, l_boundaryR, l_boundaryB, l_boundaryT);
    }
    else
    {
      l_waveProp = new tsunami_lab::patches::WavePropagationCUDA(l_nx, l_ny, l_boundaryL, l_boundaryR, l_boundaryB, l_boundaryT);
    }

    // set up solver
#pragma omp parallel for
    for (tsunami_lab::t_idx l_cy = 0; l_cy < l_ny; l_cy++)
      for (tsunami_lab::t_idx l_cx = 0; l_cx < l_nx; l_cx++)
      {
        l_waveProp->setHeight(l_cx,
                              l_cy,
                              l_h[l_cx + l_cy * l_nx]);

        l_waveProp->setMomentumX(l_cx,
                                 l_cy,
                                 l_hu[l_cx + l_cy * l_nx]);

        l_waveProp->setMomentumY(l_cx,
                                 l_cy,
                                 l_hv[l_cx + l_cy * l_nx]);

        l_waveProp->setBathymetry(l_cx,
                                  l_cy,
                                  l_b[l_cx + l_cy * l_nx]);
      }

    // delete all newer station outputs
    if (std::filesystem::exists("stations"))
    {
      rapidcsv::Document l_doc;
      tsunami_lab::t_real l_simTime = l_timeStep * (0.45 * (l_width / l_nx) / (std::sqrt(9.81 * l_hMax)));
      for (const auto &entry : std::filesystem::directory_iterator("stations"))
      {
        std::string l_filePath{entry.path().u8string()};
        tsunami_lab::io::Csv::openCSV(l_filePath, l_doc, true);
        int l_rows = l_doc.GetRowCount();
        for (int i = l_rows - 1; i >= 0; --i)
        {
          if (l_doc.GetCell<tsunami_lab::t_real>(0, i) > l_simTime)
            l_doc.RemoveRow(i);
          else
            break;
        }
        l_doc.Save();
      }
    }

    delete[] l_b;
    delete[] l_h;
    delete[] l_hu;
    delete[] l_hv;
  }
  else
  {
    // missing n_cells_x or getting -key as last argument (f.E. -h)
    if ((i_argc < 2) || (i_argv[i_argc - 1] == 0) || (i_argv[i_argc - 1][0] == '-'))
    {
      // removed invalid number of arguments message for -h option
      std::cerr << "usage:" << std::endl;
      std::cerr << "  ./build/tsunami_lab [-s solver] [-u setup] [-b boundary] [-r stations] [-o outputType] [-f frames] [-t maxtime] [-k size] [-i] n_cells_x" << std::endl;
      std::cerr << "  more info at https://tsunami-lab.readthedocs.io/en/latest/" << std::endl;
      return EXIT_FAILURE;
    }

    // set cell size
    l_nx = atoi(i_argv[i_argc - 1]);
    if (l_nx < 1)
    {
      std::cerr << "invalid number of cells" << std::endl;
      return EXIT_FAILURE;
    }

    // get command line arguments
    opterr = 0; // disable error messages of getopt
    int opt;

    while ((opt = getopt(i_argc, i_argv, "u:s:b:r:o:f:t:k:ic")) != -1)
    {
      switch (opt)
      {
      // solver
      case 's':
      {
        std::string l_arg(optarg);
        std::transform(l_arg.begin(), l_arg.end(), l_arg.begin(), ::toupper);
        if (l_arg == "ROE")
        {
          l_useFwave = false;
        }
        else if (l_arg == "FWAVE")
        {
          l_useFwave = true;
        }
        else
        {
          std::cerr << "unknown solver " << l_arg << std::endl;
          return EXIT_FAILURE;
        }
        break;
      }
      // setup
      case 'u':
      {
        std::string l_arg(optarg);

        // split string by space
        std::stringstream l_stream(l_arg);
        std::string l_setupName, l_arg1Str, l_arg2Str, l_arg3Str, l_arg4Str, l_arg5Str;
        l_stream >> l_setupName >> l_arg1Str >> l_arg2Str >> l_arg3Str >> l_arg4Str >> l_arg5Str;

        // convert to upper case
        std::transform(l_setupName.begin(), l_setupName.end(), l_setupName.begin(), ::toupper);

        // 'Dambreak1d h_l h_r' setup
        if (l_setupName == "DAMBREAK1D")
        {
          double l_arg1 = std::stof(l_arg1Str);
          double l_arg2 = std::stof(l_arg2Str);
          std::cout << "  using DamBreak1d(" << l_arg1 << "," << l_arg2 << ",5) setup" << std::endl;
          l_setup = new tsunami_lab::setups::DamBreak1d(l_arg1,
                                                        l_arg2,
                                                        5);
        }
        else if (l_setupName == "DAMBREAK2D")
        {
          std::cout << "  using DamBreak2d() setup" << std::endl;
          l_setup = new tsunami_lab::setups::DamBreak2d();
          l_width = 100;
          l_ny = l_nx; // square domain
          l_xOffset = -50;
          l_yOffset = -50;
          l_endTime = 20;
        }
        // 'RareRare1d h hu' setup
        else if (l_setupName == "RARERARE1D")
        {
          double l_arg1 = std::stof(l_arg1Str);
          double l_arg2 = std::stof(l_arg2Str);
          std::cout << "  using RareRare1d(" << l_arg1 << "," << l_arg2 << ",5) setup" << std::endl;
          l_setup = new tsunami_lab::setups::RareRare1d(l_arg1,
                                                        l_arg2,
                                                        5);
        }
        // 'ShockShock1d h hu' setup
        else if (l_setupName == "SHOCKSHOCK1D")
        {
          double l_arg1 = std::stof(l_arg1Str);
          double l_arg2 = std::stof(l_arg2Str);
          std::cout << "  using ShockShock1d(" << l_arg1 << "," << l_arg2 << ",5) setup" << std::endl;
          l_setup = new tsunami_lab::setups::ShockShock1d(l_arg1,
                                                          l_arg2,
                                                          5);
        }
        // 'Custom1d h_l h_r hu_l hu_r middle' setup
        else if (l_setupName == "CUSTOM1D")
        {
          double l_arg1 = std::stof(l_arg1Str);
          double l_arg2 = std::stof(l_arg2Str);
          double l_arg3 = std::stof(l_arg3Str);
          double l_arg4 = std::stof(l_arg4Str);
          double l_arg5 = std::stof(l_arg5Str);
          std::cout << "  using Custom1d(" << l_arg1 << "," << l_arg2 << "," << l_arg3 << "," << l_arg4 << "," << l_arg5 << ") setup" << std::endl;
          l_setup = new tsunami_lab::setups::Custom1d(l_arg1,
                                                      l_arg2,
                                                      l_arg3,
                                                      l_arg4,
                                                      l_arg5);
        }
        // 'Supercrit1d' setup
        else if (l_setupName == "SUPERCRIT1D")
        {
          l_width = 25.0;  // 25 m domain
          l_endTime = 200; // 200 s simulation time
          std::cout << "  using Supercritical1d() setup" << std::endl;
          l_setup = new tsunami_lab::setups::Supercritical1d();
        }
        // 'Subcrit1d' setup
        else if (l_setupName == "SUBCRIT1D")
        {
          l_width = 25.0;  // 25 m domain
          l_endTime = 200; // 200 s simulation time
          std::cout << "  using Subcritical1d() setup" << std::endl;
          l_setup = new tsunami_lab::setups::Subcritical1d();
        }
        // 'Tsunami1d pathToCsv time' setup
        else if (l_setupName == "TSUNAMI1D")
        {
          // assumptions: headerless csv, 4 columns, 3rd being length along path and 4th being height
          std::string l_filePath = l_arg1Str;
          rapidcsv::Document l_doc;

          tsunami_lab::io::Csv::openCSV(l_filePath, l_doc, false);

          l_setup = new tsunami_lab::setups::TsunamiEvent1d(l_doc);
          l_width = 250 * l_doc.GetRowCount();
          l_endTime = stof(l_arg2Str);
          std::cout << "  using Tsunami1d(" << l_filePath << ") setup" << std::endl;
        }
        // 'ArtificialTsunami2d time' setup
        else if (l_setupName == "ARTIFICIALTSUNAMI2D")
        {
          l_setup = new tsunami_lab::setups::ArtificialTsunami2d();
          l_width = 10000;
          l_ny = l_nx; // square domain
          l_xOffset = -5000;
          l_yOffset = -5000;
          l_endTime = stof(l_arg1Str);
          std::cout << "  using ArtificialTsunami2d(" << l_endTime << ") setup" << std::endl;
        }
        // 'Tsunami2d pathToDisplacement pathToBathymetry time' setup
        else if (l_setupName == "TSUNAMI2D")
        {
          tsunami_lab::t_real l_height = -1;
          l_setup = new tsunami_lab::setups::TsunamiEvent2d(l_arg1Str.data(), l_arg2Str.data(), &l_width, &l_height, &l_xOffset, &l_yOffset);
          l_nx = l_width / l_nx; // l_nx is the resolution in meter in this case
          l_ny = l_nx * l_height / l_width;
          l_endTime = stof(l_arg3Str);
          std::cout << "  using Tsunami2d(" << l_arg1Str << "," << l_arg2Str << "," << l_endTime << ") setup" << std::endl;
        }
        // unknown setup
        else
        {
          std::cerr << "unknown setup " << l_setupName << std::endl;
          return EXIT_FAILURE;
        }
        break;
      }
      // boundary
      case 'b':
      {
        std::string l_arg(optarg);

        // convert to upper case
        std::transform(l_arg.begin(), l_arg.end(), l_arg.begin(), ::toupper);

        // split string by space
        std::stringstream l_stream(l_arg);
        std::string l_boundaryLName, l_boundaryRName, l_boundaryBName, l_boundaryTName;
        l_stream >> l_boundaryLName >> l_boundaryRName >> l_boundaryBName >> l_boundaryTName;

        std::cout << "  using boundary conditions "
                  << "  left: " << l_boundaryLName << "  right: " << l_boundaryRName << std::endl;
        std::cout << "                            "
                  << "bottom: " << l_boundaryBName << "    top: " << l_boundaryTName << std::endl;

        // convert to t_boundary
        getBoundary(l_boundaryLName, &l_boundaryL);
        getBoundary(l_boundaryRName, &l_boundaryR);
        getBoundary(l_boundaryBName, &l_boundaryB);
        getBoundary(l_boundaryTName, &l_boundaryT);
        break;
      }
      // stations
      case 'r':
      {
        l_stationsPath = optarg;
        std::cout << "  using stations file at " << l_stationsPath << std::endl;
        break;
      }
      // output
      case 'o':
      {
        std::string l_outputName(optarg);

        // convert to upper case
        std::transform(l_outputName.begin(), l_outputName.end(), l_outputName.begin(), ::toupper);

        if (l_outputName == "CSV")
        {
          l_useNetCdf = false;
        }
        else if (l_outputName == "NETCDF")
        {
          l_useNetCdf = true;
        }
        else
        {
          std::cerr << "unknown output " << l_outputName << std::endl;
          return EXIT_FAILURE;
        }

        break;
      }
      // frames
      case 'f':
      {
        l_nFrames = atoi(optarg);
        break;
      }
      // maxtime
      case 't':
      {
        l_maxHours = atoi(optarg);
        break;
      }
      // coarse output
      case 'k':
      {
        l_k = atoi(optarg);
        if (l_k < 1)
        {
          std::cerr << "invalid coarse output " << l_k << std::endl;
          return EXIT_FAILURE;
        }
        break;
      }
      // benchmarking no fileIO
      case 'i':
      {
        l_useFileIO = false;
        break;
      }
      // use CUDA
      case 'c':
      {
        l_useCuda = true;
        break;
      }
      // unknown option
      case '?':
      {
        std::cerr << "unknown option: " << char(optopt) << std::endl;
        break;
      }
      }
    }

    if (l_stationsPath.empty())
    {
      std::cout << "  using stations file at src/data/stations.json" << std::endl;
      l_stationsPath = "src/data/stations.json";
    }

    if (l_setup == nullptr)
    {
      std::cout << "  using DamBreak2d() setup" << std::endl;
      l_setup = new tsunami_lab::setups::DamBreak2d();
      l_width = 100;
      l_ny = l_nx; // square domain
      l_xOffset = -50;
      l_yOffset = -50;
      l_endTime = 20;
    }

    if (l_useNetCdf)
    {
      std::cout << "  using NetCdf output" << std::endl;
    }
    else
    {
      std::cout << "  using CSV output" << std::endl;
    }

    // 1d or 2d solver
    if (l_ny == 1)
    {
      l_waveProp = new tsunami_lab::patches::WavePropagation1d(l_nx, l_useFwave, l_boundaryL, l_boundaryR);
      l_k = 1;
      std::cout << "  using 1d solver" << std::endl;
    }
    else
    {
      if (!l_useCuda)
      {
        l_waveProp = new tsunami_lab::patches::WavePropagation2d(l_nx, l_ny, l_useFwave, l_boundaryL, l_boundaryR, l_boundaryB, l_boundaryT);
        std::cout << "  using 2d solver" << std::endl;
      }
      else
      {
        l_waveProp = new tsunami_lab::patches::WavePropagationCUDA(l_nx, l_ny, l_boundaryL, l_boundaryR, l_boundaryB, l_boundaryT);
        std::cout << "  using CUDA solver" << std::endl;
      }
    }
  }

  // FWave or Roe Solver
  if (l_useFwave == false)
  {
    std::cout << "  using Roe solver" << std::endl;
  }
  else
  {
    std::cout << "  using FWave solver" << std::endl;
  }

  // calculate cell size
  tsunami_lab::t_real l_dxy = l_width / l_nx;
  std::cout << "  cell size:                      " << l_dxy << " m" << std::endl;
  std::cout << "  frames output:                  " << l_nFrames << std::endl;
  std::cout << "  width simulated:                " << l_width << " m" << std::endl;
  std::cout << "  coordinates simulated:          x e [" << l_xOffset << ", " << l_width + l_xOffset << "]" << std::endl;
  std::cout << "                                  y e [" << l_yOffset << ", " << l_width * float(l_ny) / float(l_nx) + l_yOffset << "]" << std::endl;
  std::cout << "  time simulated:                 " << l_endTime << " s" << std::endl;
  std::cout << "  number of cells in x-direction: " << l_nx << std::endl;
  std::cout << "  number of cells in y-direction: " << l_ny << std::endl;
  std::cout << "  using coarse output (netcdf):   " << l_k << "x" << l_k << " cells" << std::endl;

  if (!l_useCheckpoint)
  {
// set up solver
#pragma omp parallel for reduction(max : l_hMax)
    for (tsunami_lab::t_idx l_cy = 0; l_cy < l_ny; l_cy++)
    {
      tsunami_lab::t_real l_y = l_cy * l_dxy + l_yOffset;
      for (tsunami_lab::t_idx l_cx = 0; l_cx < l_nx; l_cx++)
      {
        tsunami_lab::t_real l_x = l_cx * l_dxy + l_xOffset;

        // get initial values of the setup
        tsunami_lab::t_real l_h = l_setup->getHeight(l_x,
                                                     l_y);
        l_hMax = std::max(l_h, l_hMax);

        tsunami_lab::t_real l_hu = l_setup->getMomentumX(l_x,
                                                         l_y);

        tsunami_lab::t_real l_hv = l_setup->getMomentumY(l_x,
                                                         l_y);

        tsunami_lab::t_real l_b = l_setup->getBathymetry(l_x,
                                                         l_y);

        // set initial values in wave propagation solver
        l_waveProp->setHeight(l_cx,
                              l_cy,
                              l_h);

        l_waveProp->setMomentumX(l_cx,
                                 l_cy,
                                 l_hu);

        l_waveProp->setMomentumY(l_cx,
                                 l_cy,
                                 l_hv);

        l_waveProp->setBathymetry(l_cx,
                                  l_cy,
                                  l_b);
      }
    }
  }

  l_waveProp->initGhostCells();

  // derive maximum wave speed in setup; the momentum is ignored
  tsunami_lab::t_real l_speedMax = std::sqrt(9.81 * l_hMax);

  // derive constant time step; changes at simulation time are ignored
  tsunami_lab::t_real l_dt = 0.45 * l_dxy / l_speedMax;

  // derive scaling for a time step
  tsunami_lab::t_real l_scaling = l_dt / l_dxy;

  // set up time and print control
  tsunami_lab::t_idx l_nTimeSteps = ceil(l_endTime / l_dt);
  tsunami_lab::t_idx l_nTimeStepsPerFrame = floor((double)l_nTimeSteps / (double)l_nFrames);
  if (l_nTimeStepsPerFrame <= 0)
  {
    l_nTimeStepsPerFrame = 1;
  }
  // init IO
  if (l_useFileIO)
  {
    if (l_useNetCdf)
    {
      l_writer = new tsunami_lab::io::NetCdf(l_dxy,
                                             l_nx,
                                             l_ny,
                                             l_waveProp->getStride(),
                                             l_waveProp->getGhostCellsX(),
                                             l_waveProp->getGhostCellsY(),
                                             l_xOffset,
                                             l_yOffset,
                                             l_k,
                                             l_waveProp->getBathymetry(),
                                             l_useCheckpoint);
    }
    else
    {
      l_writer = new tsunami_lab::io::Csv(l_dxy,
                                          l_nx,
                                          l_ny,
                                          l_waveProp->getStride(),
                                          l_waveProp->getGhostCellsX(),
                                          l_waveProp->getGhostCellsY(),
                                          l_xOffset,
                                          l_yOffset,
                                          l_k,
                                          l_waveProp->getBathymetry(),
                                          l_useCheckpoint);
    }
    l_stations = new tsunami_lab::io::Stations(l_stationsPath,
                                               l_dxy,
                                               l_nx,
                                               l_ny,
                                               l_waveProp->getStride(),
                                               l_waveProp->getGhostCellsX(),
                                               l_waveProp->getGhostCellsY(),
                                               l_xOffset,
                                               l_yOffset,
                                               l_waveProp->getBathymetry(),
                                               l_useCheckpoint);
  }

  tsunami_lab::t_idx l_nStepsPerStation = 1;
  if (l_useFileIO && l_stations->hasStations() && floor((double)l_stations->getT() / (double)l_dt) > 1)
  {
    l_nStepsPerStation = floor((double)l_stations->getT() / (double)l_dt);
  }

  std::cout << "  time step:                      " << l_dt << " s" << std::endl;
  std::cout << "  number of time steps:           " << l_nTimeSteps << std::endl;
  std::cout << "  time steps per frame:           " << l_nTimeStepsPerFrame << std::endl;
  std::cout << "  time steps per station:         " << l_nStepsPerStation << std::endl;
  std::cout << "  time per frame (approx.):       " << l_nTimeStepsPerFrame * l_dt << " s" << std::endl;
  std::cout << "  maximum runtime:                " << l_maxHours << " h" << std::endl;
  std::cout << "entering time loop" << std::endl;

  auto l_timeSetup = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds l_duration_write = std::chrono::nanoseconds::zero();
  std::chrono::nanoseconds l_duration_checkpoint = std::chrono::nanoseconds::zero();
  // writing 1 checkpoint per hour
  int l_nOutCheckpoint = 1;

  // iterate over time
  for (; l_timeStep < l_nTimeSteps; l_timeStep++)
  {
    if (l_useFileIO && l_timeStep % l_nTimeStepsPerFrame == 0)
    {
      tsunami_lab::t_real l_simTime = l_dt * l_timeStep;
      std::cout << "  simulation time / #time steps: "
                << l_simTime << " / " << l_timeStep << std::endl;

      l_waveProp->prepareDataAccess();
      auto l_writeStart = std::chrono::high_resolution_clock::now();
      l_writer->write(l_waveProp->getHeight(),
                      l_waveProp->getMomentumX(),
                      l_waveProp->getMomentumY(),
                      l_simTime,
                      l_nOut);
      l_nOut++;

      auto l_WriteEnd = std::chrono::high_resolution_clock::now();
      l_duration_write += l_WriteEnd - l_writeStart;

      // stop if current time exceeds max_hours
      auto l_elapsed = l_WriteEnd - l_start;
      if (l_elapsed >= std::chrono::hours(l_maxHours))
      {
        std::cout << "  maximum time exceeded, exiting" << std::endl;
        break;
      }
      // write checkpoint every hour (only 2D, netCdf)
      else if (l_ny > 1 && l_useNetCdf && l_elapsed >= std::chrono::hours(l_nOutCheckpoint))
      {
        std::cout << "  writing checkpoint" << std::endl;
        tsunami_lab::io::NetCdf::writeCheckpoint(l_nx,
                                                 l_ny,
                                                 l_waveProp->getStride(),
                                                 l_waveProp->getGhostCellsX(),
                                                 l_waveProp->getGhostCellsY(),
                                                 l_useFwave,
                                                 l_useCuda,
                                                 l_boundaryL,
                                                 l_boundaryR,
                                                 l_boundaryB,
                                                 l_boundaryT,
                                                 l_endTime,
                                                 l_width,
                                                 l_xOffset,
                                                 l_yOffset,
                                                 l_hMax,
                                                 l_stations->getPath(),
                                                 l_nFrames,
                                                 l_k,
                                                 l_timeStep,
                                                 l_nOut,
                                                 l_maxHours,
                                                 l_waveProp->getBathymetry(),
                                                 l_waveProp->getHeight(),
                                                 l_waveProp->getMomentumX(),
                                                 l_waveProp->getMomentumY());
        ++l_nOutCheckpoint;

        // delete oldest checkpoint if at least 3 exist
        std::vector<std::string> l_checkpoints = {};
        for (const auto &entry : std::filesystem::directory_iterator("checkpoints"))
        {
          l_checkpoints.push_back(entry.path());
        }
        std::sort(l_checkpoints.begin(), l_checkpoints.end());
        if (l_checkpoints.size() > 2)
        {
          std::filesystem::remove(l_checkpoints.front());
        }
        auto l_checkpointEnd = std::chrono::high_resolution_clock::now();
        l_duration_checkpoint += l_checkpointEnd - l_WriteEnd;
      }
    }

    if (l_useFileIO && l_stations->hasStations() && l_timeStep % l_nStepsPerStation == 0)
    {
      tsunami_lab::t_real l_simTime = l_dt * l_timeStep;
      l_waveProp->prepareDataAccess();
      l_stations->write(l_simTime,
                        l_waveProp->getHeight(),
                        l_waveProp->getMomentumX(),
                        l_waveProp->getMomentumY());
    }

    l_waveProp->timeStep(l_scaling);
  }

  std::cout << "finished time loop" << std::endl;
  auto l_end = std::chrono::high_resolution_clock::now();
  auto l_duration_total = l_end - l_start;
  printTime(l_duration_total, "total time");
  auto l_duration_setup = l_timeSetup - l_start;
  printTime(l_duration_setup, "setup time");
  auto l_duration_loop = l_end - l_timeSetup;
  auto l_duration_calc = l_duration_loop - l_duration_write - l_duration_checkpoint;
  printTime(l_duration_calc, "calc time ");
  printTime(l_duration_write, "write time");
  printTime(l_duration_checkpoint, "checkpoint time");
  std::cout << "calc time per cell and iteration: " << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(l_duration_calc).count() / (double)(l_nTimeSteps * l_nx * l_ny) << "ns" << std::endl;

  // free memory
  std::cout << "freeing memory" << std::endl;
  delete l_setup;
  delete l_waveProp;
  delete l_writer;
  delete l_stations;

  std::cout << "finished, exiting" << std::endl;
  return EXIT_SUCCESS;
}
