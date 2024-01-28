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
#include <cmath>

//! gravity
static tsunami_lab::t_real constexpr m_g = 9.80665;
//! square root of gravity
static tsunami_lab::t_real constexpr m_gSqrt = 3.131557121;

__global__ void setGhostCellsX(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hu, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_boundary i_boundaryLeft, tsunami_lab::t_boundary i_boundaryRight);
__global__ void setGhostCellsY(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_boundary i_boundaryBottom, tsunami_lab::t_boundary i_boundaryTop);
__global__ void initGhostCellsCuda(tsunami_lab::t_real *io_b, tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hu, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_boundary i_boundaryLeft, tsunami_lab::t_boundary i_boundaryRight, tsunami_lab::t_boundary i_boundaryBottom, tsunami_lab::t_boundary i_boundaryTop);
__global__ void netUpdatesX(tsunami_lab::t_real *o_h, tsunami_lab::t_real *o_hu, tsunami_lab::t_real *i_hTemp,tsunami_lab::t_real * i_huvTemp, tsunami_lab::t_real *i_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_real i_scaling);
__global__ void netUpdatesY(tsunami_lab::t_real *o_h, tsunami_lab::t_real *o_hv, tsunami_lab::t_real *i_hTemp,tsunami_lab::t_real * i_huvTemp, tsunami_lab::t_real *i_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_real i_scaling);
__device__ void netUpdatesCUDA(tsunami_lab::t_real i_hL,tsunami_lab::t_real i_hR,tsunami_lab::t_real i_huL,tsunami_lab::t_real i_huR,tsunami_lab::t_real i_bL,tsunami_lab::t_real i_bR,tsunami_lab::t_real o_netUpdateL[2],tsunami_lab::t_real o_netUpdateR[2]);


tsunami_lab::t_idx tsunami_lab::patches::WavePropagationCUDA::getCoord(t_idx i_x, t_idx i_y)
{
    return i_x + i_y * (m_nCellsx + 2);
}

tsunami_lab::patches::WavePropagationCUDA::WavePropagationCUDA(t_idx i_nCellsx,
                                                               t_idx i_nCellsy,
                                                               t_boundary i_boundaryLeft,
                                                               t_boundary i_boundaryRight,
                                                               t_boundary i_boundaryBottom,
                                                               t_boundary i_boundaryTop) : m_nCellsx(i_nCellsx),
                                                                                           m_nCellsy(i_nCellsy),
                                                                                           m_boundaryLeft(i_boundaryLeft),
                                                                                           m_boundaryRight(i_boundaryRight),
                                                                                           m_boundaryBottom(i_boundaryBottom),
                                                                                           m_boundaryTop(i_boundaryTop)
{
    // allocate memory including a single ghost cell on each side (zero initialised)
    t_idx l_size = (m_nCellsx + 2) * (m_nCellsy + 2) * sizeof(float);
    cudaMalloc(&m_h, l_size);
    cudaMalloc(&m_hu, l_size);
    cudaMalloc(&m_hv, l_size);
    cudaMalloc(&m_hTemp, l_size);
    cudaMalloc(&m_huvTemp, l_size);
    cudaMalloc(&m_b, l_size);
    cudaMemset(m_h, 0, l_size);
    cudaMemset(m_hu, 0, l_size);
    cudaMemset(m_hv, 0, l_size);
    cudaMemset(m_hTemp, 0, l_size);
    cudaMemset(m_huvTemp, 0, l_size);
    cudaMemset(m_b, 0, l_size);

    m_h_host = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
    m_hu_host = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
    m_hv_host = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
    m_b_host = new t_real[(m_nCellsx + 2) * (m_nCellsy + 2)];
}

tsunami_lab::patches::WavePropagationCUDA::~WavePropagationCUDA()
{
    cudaFree(m_h);
    cudaFree(m_hu);
    cudaFree(m_hv);
    cudaFree(m_hTemp);
    cudaFree(m_huvTemp);
    cudaFree(m_b);
    delete[] m_h_host;
    delete[] m_hu_host;
    delete[] m_hv_host;
    delete[] m_b_host;
}

void tsunami_lab::patches::WavePropagationCUDA::timeStep(t_real i_scaling)
{
    dim3 l_blockSize(16, 16);
    dim3 l_numBlock((m_nCellsx+2-1)/l_blockSize.x+1, (m_nCellsy+2-1)/l_blockSize.y+1);

    setGhostCellsX<<<l_numBlock,l_blockSize>>>(m_h, m_hu, m_nCellsx, m_nCellsy, m_boundaryLeft, m_boundaryRight);

    cudaMemcpy(m_hTemp, m_h, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_huvTemp, m_hu, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToDevice);
    netUpdatesX<<<l_numBlock,l_blockSize>>>(m_h, m_hu, m_hTemp, m_huvTemp, m_b, m_nCellsx, m_nCellsy, i_scaling);

    setGhostCellsY<<<l_numBlock,l_blockSize>>>(m_h, m_hv, m_nCellsx, m_nCellsy, m_boundaryBottom, m_boundaryTop);

    cudaMemcpy(m_hTemp, m_h, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_huvTemp, m_hv, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToDevice);
    netUpdatesY<<<l_numBlock,l_blockSize>>>(m_h, m_hv, m_hTemp, m_huvTemp, m_b, m_nCellsx, m_nCellsy, i_scaling);
}

__global__ void netUpdatesY(tsunami_lab::t_real *o_h, tsunami_lab::t_real *o_hv, tsunami_lab::t_real *i_hTemp,tsunami_lab::t_real *i_huvTemp, tsunami_lab::t_real *i_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_real i_scaling)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_x > i_nx + 1 || l_y > i_ny + 1)
    {
        return;
    }
    // determine top and bottom cell-id
    tsunami_lab::t_idx l_ceB = l_x + l_y * (i_nx + 2);
    tsunami_lab::t_idx l_ceT = l_x + (l_y + 1) * (i_nx + 2);

    // compute net-updates
    tsunami_lab::t_real l_netUpdates[2][2];

    netUpdatesCUDA(i_hTemp[l_ceB],
                   i_hTemp[l_ceT],
                   i_huvTemp[l_ceB],
                   i_huvTemp[l_ceT],
                   i_b[l_ceB],
                   i_b[l_ceT],
                   l_netUpdates[0],
                   l_netUpdates[1]);

    // update the cells' quantities
    atomicAdd(&o_h[l_ceB], -i_scaling * l_netUpdates[0][0]);
    atomicAdd(&o_hv[l_ceB], -i_scaling * l_netUpdates[0][1]);

    atomicAdd(&o_h[l_ceT], -i_scaling * l_netUpdates[1][0]);
    atomicAdd(&o_hv[l_ceT], -i_scaling * l_netUpdates[1][1]);
}


__global__ void netUpdatesX(tsunami_lab::t_real *o_h, tsunami_lab::t_real *o_hu, tsunami_lab::t_real *i_hTemp,tsunami_lab::t_real * i_huvTemp, tsunami_lab::t_real *i_b, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_real i_scaling)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_x > i_nx + 1 || l_y > i_ny + 1)
    {
        return;
    }
    
    // determine left and right cell-id
    tsunami_lab::t_idx l_ceL = l_x + l_y * (i_nx + 2);
    tsunami_lab::t_idx l_ceR = l_x + 1 + l_y * (i_nx + 2);

    // compute net-updates
    tsunami_lab::t_real l_netUpdates[2][2];

    netUpdatesCUDA(i_hTemp[l_ceL],
                   i_hTemp[l_ceR],
                   i_huvTemp[l_ceL],
                   i_huvTemp[l_ceR],
                   i_b[l_ceL],
                   i_b[l_ceR],
                   l_netUpdates[0],
                   l_netUpdates[1]);

    // update the cells' quantities
    atomicAdd(&o_h[l_ceL], -i_scaling * l_netUpdates[0][0]);
    atomicAdd(&o_hu[l_ceL], -i_scaling * l_netUpdates[0][1]);

    atomicAdd(&o_h[l_ceR], -i_scaling * l_netUpdates[1][0]);
    atomicAdd(&o_hu[l_ceR], -i_scaling * l_netUpdates[1][1]);
}

__global__ void setGhostCellsX(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hu, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_boundary i_boundaryLeft, tsunami_lab::t_boundary i_boundaryRight)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (l_x > i_nx + 1 || l_y > i_ny + 1)
    {
        return;
    }

    if (l_x == 0 && i_boundaryLeft == tsunami_lab::t_boundary::OPEN)
    {
        io_h[(i_nx+2) * l_y] = io_h[1 + (i_nx+2) * l_y];
        io_hu[(i_nx+2) * l_y] = io_hu[1 + (i_nx+2) * l_y];
    }
    else if (l_x == i_nx + 1 && i_boundaryRight == tsunami_lab::t_boundary::OPEN)
    {
        io_h[l_x + (i_nx+2) * l_y] = io_h[l_x - 1 + (i_nx+2) * l_y];
        io_hu[l_x + (i_nx+2) * l_y] = io_hu[l_x - 1 + (i_nx+2) * l_y];
    }
}

__global__ void setGhostCellsY(tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_boundary i_boundaryBottom, tsunami_lab::t_boundary i_boundaryTop)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (l_x > i_nx + 1 || l_y > i_ny + 1)
    {
        return;
    }

    if (l_y == 0 && i_boundaryBottom == tsunami_lab::t_boundary::OPEN)
    {
        io_h[l_x] = io_h[l_x + (i_nx+2)];
        io_hv[l_x] = io_hv[l_x + (i_nx+2)];
    }
    else if (l_y == i_ny + 1 && i_boundaryTop == tsunami_lab::t_boundary::OPEN)
    {
        io_h[l_x + (i_nx+2) * l_y] = io_h[l_x + (i_nx+2) * (l_y - 1)];
        io_hv[l_x + (i_nx+2) * l_y] = io_hv[l_x + (i_nx+2) * (l_y - 1)];
    }
}

void tsunami_lab::patches::WavePropagationCUDA::initGhostCells()
{
    // copy host data to device
    cudaMemcpy(m_h, m_h_host, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_hu, m_hu_host, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_hv, m_hv_host, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_b, m_b_host, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyHostToDevice);

    // set ghost cells
    dim3 l_blockSize(16,16);
    dim3 l_numBlock((m_nCellsx+2-1)/l_blockSize.x+1, (m_nCellsy+2-1)/l_blockSize.y+1);
    initGhostCellsCuda<<<l_numBlock,l_blockSize>>>(m_b, m_h, m_hu, m_hv, m_nCellsx, m_nCellsy, m_boundaryLeft, m_boundaryRight, m_boundaryBottom, m_boundaryTop);
}

void tsunami_lab::patches::WavePropagationCUDA::prepareDataAccess()
{
    cudaDeviceSynchronize();
    cudaMemcpy(m_h_host, m_h, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_hu_host, m_hu, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_hv_host, m_hv, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void initGhostCellsCuda(tsunami_lab::t_real *io_b, tsunami_lab::t_real *io_h, tsunami_lab::t_real *io_hu, tsunami_lab::t_real *io_hv, tsunami_lab::t_idx i_nx, tsunami_lab::t_idx i_ny, tsunami_lab::t_boundary i_boundaryLeft, tsunami_lab::t_boundary i_boundaryRight, tsunami_lab::t_boundary i_boundaryBottom, tsunami_lab::t_boundary i_boundaryTop)
{
    tsunami_lab::t_idx l_x = blockIdx.x * blockDim.x + threadIdx.x;
    tsunami_lab::t_idx l_y = blockIdx.y * blockDim.y + threadIdx.y;
    // only affect cells in grid
    if (l_x > i_nx + 1 || l_y > i_ny + 1)
    {
        return;
    }

    if (l_x == 0)
    {
        if(i_boundaryLeft == tsunami_lab::t_boundary::OPEN)
        {
            io_b[(i_nx+2) * l_y] = io_b[1 + (i_nx+2) * l_y];
        }
        else if(i_boundaryLeft == tsunami_lab::t_boundary::WALL)
        {
            io_b[(i_nx+2) * l_y] = 20;
            io_h[(i_nx+2) * l_y] = 0;
            io_hu[(i_nx+2) * l_y] = 0;
            io_hv[(i_nx+2) * l_y] = 0;
        }
    }
    else if (l_x == i_nx + 1)
    {
        if(i_boundaryRight == tsunami_lab::t_boundary::OPEN)
        {
            io_b[l_x + (i_nx+2) * l_y] = io_b[i_nx + (i_nx+2) * l_y];
        }
        else if(i_boundaryRight == tsunami_lab::t_boundary::WALL)
        {
            io_b[l_x + (i_nx+2) * l_y] = 20;
            io_h[l_x + (i_nx+2) * l_y] = 0;
            io_hu[l_x + (i_nx+2) * l_y] = 0;
            io_hv[l_x + (i_nx+2) * l_y] = 0;
        }
    }
    else if (l_y == 0)
    {
        if(i_boundaryBottom == tsunami_lab::t_boundary::OPEN)
        {
            io_b[l_x] = io_b[l_x + (i_nx+2)];
        }
        else if(i_boundaryBottom == tsunami_lab::t_boundary::WALL)
        {
            io_b[l_x] = 20;
            io_h[l_x] = 0;
            io_hu[l_x] = 0;
            io_hv[l_x] = 0;
        }
    }
    else if (l_y == i_ny + 1)
    {
        if(i_boundaryTop == tsunami_lab::t_boundary::OPEN)
        {
            io_b[l_x + (i_nx+2) * l_y] = io_b[l_x + (i_nx+2) * (l_y - 1)];
        }
        else if(i_boundaryTop == tsunami_lab::t_boundary::WALL)
        {
            io_b[l_x + (i_nx+2) * l_y] = 20;
            io_h[l_x + (i_nx+2) * l_y] = 0;
            io_hu[l_x + (i_nx+2) * l_y] = 0;
            io_hv[l_x + (i_nx+2) * l_y] = 0;
        }
    }
}

__device__ void waveSpeeds(tsunami_lab::t_real i_hL,
                           tsunami_lab::t_real i_hR,
                           tsunami_lab::t_real i_uL,
                           tsunami_lab::t_real i_uR,
                           tsunami_lab::t_real &o_waveSpeedL,
                           tsunami_lab::t_real &o_waveSpeedR)
{
    // pre-compute square-root ops
    tsunami_lab::t_real l_hSqrtL = std::sqrt(i_hL);
    tsunami_lab::t_real l_hSqrtR = std::sqrt(i_hR);

    // compute Roe averages
    tsunami_lab::t_real l_hRoe = tsunami_lab::t_real(0.5) * (i_hL + i_hR);
    tsunami_lab::t_real l_uRoe = l_hSqrtL * i_uL + l_hSqrtR * i_uR;
    l_uRoe /= l_hSqrtL + l_hSqrtR;

    // compute wave speeds
    tsunami_lab::t_real l_ghSqrtRoe = m_gSqrt * std::sqrt(l_hRoe);
    o_waveSpeedL = l_uRoe - l_ghSqrtRoe;
    o_waveSpeedR = l_uRoe + l_ghSqrtRoe;
}

__device__ void flux(tsunami_lab::t_real i_h,
                     tsunami_lab::t_real i_hu,
                     tsunami_lab::t_real &o_flux0,
                     tsunami_lab::t_real &o_flux1)
{
    // f(q) = [hu, h*u^2 + 1/2*g*h^2]
    o_flux0 = i_hu;
    o_flux1 = i_hu * i_hu / i_h + tsunami_lab::t_real(0.5) * m_g * i_h * i_h;
}

__device__ void deltaXPsi(tsunami_lab::t_real i_bL,
                          tsunami_lab::t_real i_bR,
                          tsunami_lab::t_real i_hL,
                          tsunami_lab::t_real i_hR,
                          tsunami_lab::t_real &o_deltaXPsi)
{
    // compute deltaXPsi
    o_deltaXPsi = -m_g * (i_bR - i_bL) * (i_hL + i_hR) / 2;
}

__device__ void waveStrengths(tsunami_lab::t_real i_hL,
                              tsunami_lab::t_real i_hR,
                              tsunami_lab::t_real i_huL,
                              tsunami_lab::t_real i_huR,
                              tsunami_lab::t_real i_bL,
                              tsunami_lab::t_real i_bR,
                              tsunami_lab::t_real i_waveSpeedL,
                              tsunami_lab::t_real i_waveSpeedR,
                              tsunami_lab::t_real &o_strengthL,
                              tsunami_lab::t_real &o_strengthR)
{
    // compute inverse of right eigenvector-matrix
    tsunami_lab::t_real l_detInv = 1 / (i_waveSpeedR - i_waveSpeedL);

    tsunami_lab::t_real l_rInv[2][2] = {0};
    l_rInv[0][0] = l_detInv * i_waveSpeedR;
    l_rInv[0][1] = -l_detInv;
    l_rInv[1][0] = -l_detInv * i_waveSpeedL;
    l_rInv[1][1] = l_detInv;

    // calculating the fluxes
    tsunami_lab::t_real l_flux0L = 0;
    tsunami_lab::t_real l_flux1L = 0;
    tsunami_lab::t_real l_flux0R = 0;
    tsunami_lab::t_real l_flux1R = 0;
    tsunami_lab::t_real l_deltaXPsi = 0;

    flux(i_hL, i_huL, l_flux0L, l_flux1L);
    flux(i_hR, i_huR, l_flux0R, l_flux1R);
    deltaXPsi(i_bL, i_bR, i_hL, i_hR, l_deltaXPsi);

    // compute jump in fluxes
    tsunami_lab::t_real l_flux0Jump = l_flux0R - l_flux0L;
    tsunami_lab::t_real l_flux1Jump = l_flux1R - l_flux1L - l_deltaXPsi;

    // compute wave strengths
    o_strengthL = l_rInv[0][0] * l_flux0Jump;
    o_strengthL += l_rInv[0][1] * l_flux1Jump;

    o_strengthR = l_rInv[1][0] * l_flux0Jump;
    o_strengthR += l_rInv[1][1] * l_flux1Jump;
}

__device__ void netUpdatesCUDA(tsunami_lab::t_real i_hL,
                               tsunami_lab::t_real i_hR,
                               tsunami_lab::t_real i_huL,
                               tsunami_lab::t_real i_huR,
                               tsunami_lab::t_real i_bL,
                               tsunami_lab::t_real i_bR,
                               tsunami_lab::t_real o_netUpdateL[2],
                               tsunami_lab::t_real o_netUpdateR[2])
{
    // initialize net-updates
    o_netUpdateL[0] = 0;
    o_netUpdateL[1] = 0;
    o_netUpdateR[0] = 0;
    o_netUpdateR[1] = 0;
    tsunami_lab::t_real temp[2] = {};

    // if only left side is dry, apply reflecting boundary condition
    if (i_hL <= 0)
    {
        // if both dry do nothing
        if (i_hR <= 0)
        {
            return;
        }
        i_hL = i_hR;
        i_huL = -i_huR;
        i_bL = i_bR;
        // unhook o_netUpdateL from data
        o_netUpdateL = temp;
    } // if only right side is dry, apply reflecting boundary condition
    else if (i_hR <= 0)
    {
        i_hR = i_hL;
        i_huR = -i_huL;
        i_bR = i_bL;
        // unhook o_netUpdateR from data
        o_netUpdateR = temp;
    }

    // compute particle velocities
    tsunami_lab::t_real l_uL = i_huL / i_hL;
    tsunami_lab::t_real l_uR = i_huR / i_hR;

    // compute wave speeds
    tsunami_lab::t_real l_sL = 0;
    tsunami_lab::t_real l_sR = 0;

    waveSpeeds(i_hL,
               i_hR,
               l_uL,
               l_uR,
               l_sL,
               l_sR);

    // compute wave strengths
    tsunami_lab::t_real l_aL = 0;
    tsunami_lab::t_real l_aR = 0;

    waveStrengths(i_hL,
                  i_hR,
                  i_huL,
                  i_huR,
                  i_bL,
                  i_bR,
                  l_sL,
                  l_sR,
                  l_aL,
                  l_aR);

    // compute waves
    tsunami_lab::t_real l_waveL[2] = {0};
    tsunami_lab::t_real l_waveR[2] = {0};

    l_waveL[0] = l_aL;
    l_waveL[1] = l_aL * l_sL;

    l_waveR[0] = l_aR;
    l_waveR[1] = l_aR * l_sR;

    // set net-updates depending on wave speeds
    for (unsigned short l_qt = 0; l_qt < 2; l_qt++)
    {
        // 1st wave
        if (l_sL < 0)
        {
            o_netUpdateL[l_qt] += l_waveL[l_qt];
        }
        else if (l_sL >= 0)
        {
            o_netUpdateR[l_qt] += l_waveL[l_qt];
        }

        // 2nd wave
        if (l_sR > 0)
        {
            o_netUpdateR[l_qt] += l_waveR[l_qt];
        }
        else if (l_sR <= 0)
        {
            o_netUpdateL[l_qt] += l_waveR[l_qt];
        }
    }
}
