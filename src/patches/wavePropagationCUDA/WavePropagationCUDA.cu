/**
 * @author Justus Dreßler (justus.dressler AT uni-jena.de)
 * @author Thorsten Kröhl (thorsten.kroehl AT uni-jena.de)
 * @author Julius Halank (julius.halank AT uni-jena.de)
 *
 * @section DESCRIPTION
 * Two-dimensional wave propagation patch.
 **/
#include "WavePropagationCUDA.h"
#include "../../solvers/roe/Roe.h"
#include "../../solvers/fWave/FWave.h"
#include <cuda.h>

__global__ void setGhostCellsX(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hu, tsunami_lab::t_idx i_nx);
__global__ void setGhostCellsY(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny);
__global__ void initGhostCellsCuda(tsunami_lab::t_real *io_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny);
__global__ void copyData(tsunami_lab::t_real* m_hTemp, tsunami_lab::t_real* m_h, tsunami_lab::t_real* m_huvTemp, tsunami_lab::t_real* m_hu, tsunami_lab::t_idx m_nCellsx, tsunami_lab::t_idx m_nCellsy);
__device__ static tsunami_lab::t_idx getCoordCuda(tsunami_lab::t_idx i_x, tsunami_lab::t_idx i_y, tsunami_lab::t_idx m_nCellsx);

tsunami_lab::patches::WavePropagationCUDA::WavePropagationCUDA(t_idx i_nCellsx,
                                                               t_idx i_nCellsy) : m_nCellsx(i_nCellsx),
                                                                                  m_nCellsy(i_nCellsy)
{
    // allocate memory including a single ghost cell on each side (zero initialised)
    t_idx l_size = (m_nCellsx + 2) * (m_nCellsy + 2) * sizeof(float);
    cudaMallocManaged(&m_h, l_size);
    cudaMallocManaged(&m_hu, l_size);
    cudaMallocManaged(&m_hv, l_size);
    cudaMallocManaged(&m_hTemp, l_size);
    cudaMallocManaged(&m_huvTemp, l_size);
    cudaMallocManaged(&m_b, l_size);
}

tsunami_lab::patches::WavePropagationCUDA::~WavePropagationCUDA()
{
    cudaFree(m_h);
    cudaFree(m_hu);
    cudaFree(m_hv);
    cudaFree(m_hTemp);
    cudaFree(m_huvTemp);
    cudaFree(m_b);
}

tsunami_lab::t_idx tsunami_lab::patches::WavePropagationCUDA::getCoord(t_idx i_x, t_idx i_y)
{
    return i_x + i_y * (m_nCellsx + 2);
}

void tsunami_lab::patches::WavePropagationCUDA::timeStep(t_real i_scaling)
{
    dim3 l_blockSize(32, 32);
    dim3 l_numBlock((m_nCellsx+2)/l_blockSize.x, (m_nCellsy+2)/l_blockSize.y);
    setGhostCellsX<<<l_numBlock,l_blockSize>>>(m_h, m_hu, m_nCellsx);
    cudaDeviceSynchronize();

// init new cell quantities
#pragma omp parallel for simd
    for (t_idx l_cy = 0; l_cy < m_nCellsy + 1; l_cy++)
        for (t_idx l_cx = 0; l_cx < m_nCellsx + 1; l_cx++)
        {
            m_hTemp[getCoord(l_cx, l_cy)] = m_h[getCoord(l_cx, l_cy)];
            m_huvTemp[getCoord(l_cx, l_cy)] = m_hu[getCoord(l_cx, l_cy)];
        }
    
// iterate over edges and update with Riemann solutions in x direction
#pragma omp parallel for
    for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
        for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
        {
            // determine left and right cell-id
            t_idx l_ceL = getCoord(l_ex, l_ey);
            t_idx l_ceR = getCoord(l_ex + 1, l_ey);

            // compute net-updates
            t_real l_netUpdates[2][2];

            solvers::FWave::netUpdates(m_hTemp[l_ceL],
                                       m_hTemp[l_ceR],
                                       m_huvTemp[l_ceL],
                                       m_huvTemp[l_ceR],
                                       m_b[l_ceL],
                                       m_b[l_ceR],
                                       l_netUpdates[0],
                                       l_netUpdates[1]);

            // update the cells' quantities
            m_h[l_ceL] -= i_scaling * l_netUpdates[0][0];
            m_hu[l_ceL] -= i_scaling * l_netUpdates[0][1];

            m_h[l_ceR] -= i_scaling * l_netUpdates[1][0];
            m_hu[l_ceR] -= i_scaling * l_netUpdates[1][1];
        }
    
    setGhostCellsY<<<l_numBlock,l_blockSize>>>(m_h, m_hv, m_nCellsx, m_nCellsy);
    copyData<<<l_numBlock,l_blockSize>>>(m_hTemp, m_h, m_huvTemp, m_hv, m_nCellsx, m_nCellsy);
    cudaDeviceSynchronize();

    // init new cell quantities
//#pragma omp parallel for simd
//    for (t_idx l_cy = 0; l_cy < m_nCellsy + 1; l_cy++)
//        for (t_idx l_cx = 0; l_cx < m_nCellsx + 1; l_cx++)
//        {
//            m_hTemp[getCoord(l_cx, l_cy)] = m_h[getCoord(l_cx, l_cy)];
//            m_huvTemp[getCoord(l_cx, l_cy)] = m_hv[getCoord(l_cx, l_cy)];
//        }

// iterate over edges and update with Riemann solutions in y direction
#pragma omp parallel for
    for (t_idx l_ex = 0; l_ex < m_nCellsx + 1; l_ex++)
        for (t_idx l_ey = 0; l_ey < m_nCellsy + 1; l_ey++)
        {
            // determine top and bottom cell-id
            t_idx l_ceB = getCoord(l_ex, l_ey);
            t_idx l_ceT = getCoord(l_ex, l_ey + 1);

            // compute net-updates
            t_real l_netUpdates[2][2];

            solvers::FWave::netUpdates(m_hTemp[l_ceB],
                                       m_hTemp[l_ceT],
                                       m_huvTemp[l_ceB],
                                       m_huvTemp[l_ceT],
                                       m_b[l_ceB],
                                       m_b[l_ceT],
                                       l_netUpdates[0],
                                       l_netUpdates[1]);

            // update the cells' quantities
            m_h[l_ceB] -= i_scaling * l_netUpdates[0][0];
            m_hv[l_ceB] -= i_scaling * l_netUpdates[0][1];

            m_h[l_ceT] -= i_scaling * l_netUpdates[1][0];
            m_hv[l_ceT] -= i_scaling * l_netUpdates[1][1];
        }
}

__global__ void setGhostCellsX(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hu, tsunami_lab::t_idx i_nx)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_x == 0)
    {
        io_h[(i_nx+2) * l_y] = io_h[1 + (i_nx+2) * l_y];
        io_hu[(i_nx+2) * l_y] = io_hu[1 + (i_nx+2) * l_y];
    }
    else if (l_x == i_nx + 1)
    {
        io_h[l_x + (i_nx+2) * l_y] = io_h[l_x - 1 + (i_nx+2) * l_y];
        io_hu[l_x + (i_nx+2) * l_y] = io_hu[l_x - 1 + (i_nx+2) * l_y];
    }
}

// __global__ void tsunami_lab::patches::WavePropagationCUDA::setGhostCellsY(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny)
__global__ void setGhostCellsY(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_y == 0)
    {
        io_h[l_x] = io_h[l_x + (i_nx+2)];
        io_hv[l_x] = io_hv[l_x + (i_nx+2)];
    }
    else if (l_y == i_ny + 1)
    {
        io_h[l_x + (i_nx+2) * l_y] = io_h[l_x + (i_nx+2) * (l_y - 1)];
        io_hv[l_x + (i_nx+2) * l_y] = io_hv[l_x + (i_nx+2) * (l_y - 1)];
    }
}

__device__ tsunami_lab::t_idx getCoordCuda(tsunami_lab::t_idx i_x, tsunami_lab::t_idx i_y, tsunami_lab::t_idx m_nCellsx)
{
    return i_x + i_y * (m_nCellsx + 2);
}

__global__ void copyData(float* m_hTemp, float* m_h, float* m_huvTemp, float* m_hv, unsigned long m_nCellsx, unsigned long m_nCellsy)
{
    int l_cx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_cy = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_cx < m_nCellsx + 1 && l_cy < m_nCellsy + 1) {
        int index = getCoordCuda(l_cx, l_cy, m_nCellsx);
        m_hTemp[index] = m_h[index];
        m_huvTemp[index] = m_hv[index];
    }
}
void tsunami_lab::patches::WavePropagationCUDA::initGhostCells()
{
    dim3 l_blockSize(32, 32);
    dim3 l_numBlock((m_nCellsx+2)/l_blockSize.x, (m_nCellsy+2)/l_blockSize.y);
    initGhostCellsCuda<<<l_numBlock,l_blockSize>>>(m_b, m_nCellsx, m_nCellsy);
    cudaDeviceSynchronize();
}

// __global__ void tsunami_lab::patches::WavePropagationCUDA::initGhostCellsCuda(tsunami_lab::t_real *io_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny)
__global__ void initGhostCellsCuda(tsunami_lab::t_real *io_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_x == 0)
    {
        io_b[l_x + (i_nx+2) * l_y] = io_b[l_x + 1 + (i_nx+2) * l_y];
    }
    else if (l_x == i_nx + 1)
    {
        io_b[l_x + (i_nx+2) * l_y] = io_b[l_x - 1 + (i_nx+2) * l_y];
    }
    else if (l_y == 0)
    {
        io_b[l_x] = io_b[l_x + (i_nx+2)];
    }
    else if (l_y == i_ny + 1)
    {
        io_b[l_x + (i_nx+2) * l_y] = io_b[l_x + (i_nx+2) * (l_y - 1)];
    }
}
