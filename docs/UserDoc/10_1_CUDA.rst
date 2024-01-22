10.1 CUDA First Meeting
=======
Links
-----

`Github Repo <https://github.com/Minutenreis/tsunami_lab>`_

`User Doc <https://tsunami-lab.readthedocs.io/en/latest/>`_

Individual Contributions
------------------------

Justus Dreßler: all members contributed equally

Thorsten Kröhl: all members contributed equally

Julius Halank: all members contributed equally


Installation Instructions and First Steps 
-----------------------------------------

https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview

and

:code:`sudo apt install nvidia-cuda-toolkit`

| So we wrote a little cuda programm that calculates the sum of two arrays to test if it works.
| To compile it we used :code:`nvcc vectorAdd.cu  -o vectorAdd" 4000`
|  

.. code:: cpp

  // computes the sum of two arrays
  #include <cstdlib>

  #include <cassert>
  #include <iostream>


  __global__ void vectorAdd(int *a, int *b, int *c, int N) {
      //calculate global thread id
      int tid = blockIdx.x * blockDim.x + threadIdx.x;

      //range check
      if (tid < N) {
          c[tid] = a[tid] + b[tid];
      }
  }

  void init_array( int *a, int N){
      for(int i=0; i<N; i++){
          a[i] = rand()%100;
      }
  }

  void verify_result(int *a, int *b, int *c, int N){
      for(int i=0; i<N; i++){
          assert(c[i] == a[i] + b[i]);
      }
      std::cout << "Success!\n";
  }

  int main(){
      // 2^20 elements
      int N = 1<<20;
      size_t bytes = N*sizeof(bytes);

      // allocate memory
      int *a, *b, *c;
      cudaMallocManaged(&a, bytes);
      cudaMallocManaged(&b, bytes);
      cudaMallocManaged(&c, bytes);

      // initialize array
      init_array(a, N);
      init_array(b, N);

      int THREADS = 256;

      // calculate block size
      // ist so weil N/Threads wäre nur 1 mit rest 1, so ist es 2
      int BLOCKS = (N + THREADS - 1)/THREADS;

      // launch kernel

      vectorAdd<<<BLOCKS, THREADS>>>(a, b, c, N);
      cudaDeviceSynchronize();

      // verify result
      verify_result(a, b, c, N);

      return 0;
  }

| The most interesting things about this small snippet is:
|  
| What are blocks and threads?
| How do i calculate them?
| And for what do i need the thread id?
|  
| So the blocks are the number of parallel processes that are running at the same time.
| The threads are the number of parallel processes that are running at the same time in one block.
| The thread id is needed to calculate the index of the array that is calculated by the thread.
|  
| To visualize this we can use the following picture:
|  

.. figure:: _static/10_cuda_indexing.png
    :width: 700

Analysis and Modification for Cuda
----------------------------------

While analyzing we noticed that we should be able to use cuda everyhwere where we used openmp.
So we just replaced the openmp pragmas and replaced the code in it with cuda kernels.

First "victim" of our replacement where the functions that calculate ghostcell-updates.
... and we ran into the first problem:

.. video:: _static/10_cuda_ghostcells_whut.mp4
   :width: 700

maybe we should use cudaDeviceSynchronize()?...

.. figure:: _static/10_cuda_ghostcells_false_index.png
    :width: 700

Seems like we have an indexing problem here.

.. video:: _static/10_cuda_ghostcells_functional.mp4
   :width: 700

... and its working!

.. code:: cpp

    dim3 l_blockSize(32, 32);
    dim3 l_numBlock((m_nCellsx+2)/l_blockSize.x, (m_nCellsy+2)/l_blockSize.y);
    setGhostCellsX<<<l_numBlock,l_blockSize>>>(m_h, m_hu, m_nCellsx);
    cudaDeviceSynchronize();

.. code:: cpp

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


Next we replaced the init new cell quantities with a cudaMemCpy instead of iterating with a custom function:

.. code:: cpp

  cudaMemcpy(m_hTemp, m_h, (m_nCellsx+2) * (m_nCellsy+2) * sizeof(float), cudaMemcpyDeviceToDevice);

Works like a charm.

Now the whole netUpdates:

Hmh. Second tsunami?...

.. video:: _static/10_cuda_Atomic_Fail.mp4
   :width: 700

Seems like the second indexing problem again, but now in our block and thread calculation.

.. code:: cpp

  dim3 l_blockSize(16,16);
  dim3 l_numBlock((m_nCellsx+2-1)/l_blockSize.x+1, (m_nCellsy+2-1)/l_blockSize.y+1);
  initGhostCellsCuda<<<l_numBlock,l_blockSize>>>(m_b, m_nCellsx, m_nCellsy);
  cudaDeviceSynchronize();


and this short snippet to limit the threads to the actual number of cells:

.. code:: cpp

    if (l_x > i_nx + 1 || l_y > i_ny + 1)
    {
        return;
    }

snippet of netUpdateX changes:

.. code:: cpp

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

and the simulation is working!

.. video:: _static/10_cuda_atomic_working.mp4
   :width: 700

It's quite slow though, at roughly 15-20 ns per cell and iteration.
That would be a setback since its slower than the openmp version (~4-6ns).

So we tried to optimize the memory a bit:

.. code:: cpp

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

Using cudaMalloc instead of cudaMallocManaged improves performance significantly.
Though to use this we have to manually send memory back and forth between host and device.
So we set the initial values on the host memory, copy it to the device together with initialising the bathymetry ghost cells and then copy it back before every write operation.

And voila: A timestep without IO costs only 0.36ns per cell and iteration or roughly 0.87ns per cell and iteration with IO.
Now the write time in NetCdf is the bottleneck.

.. code:: shell

    total time: 20s 846ms 543us 685ns
    setup time: 1s 929ms 653us 534ns
    calc time : 674ms 869us 867ns
    write time: 18s 242ms 20us 284ns
    checkpoint time: 0ns
    calc time per cell and iteration: 0.873205ns

at 4000m cell size the write time for checkpoints is roughly 27 times greater than the calculation time.
And its not getting that much better with higher resolution, at 1000m cell size it currently takes 14.36s to calculate the updates, but 2m 51.45s to write them.
So still 12 times longer write than calculation time.

We will look into this for our final assignment and at other ways to speed things up / reintroduce feature parity with the openmp version.

But this is for the next time :)

Take a final proof of our cuda version:

.. video:: _static/10_Tohoko_1000_CUDA.mp4
   :width: 700

*CUDA Version of Tohoku Simulation 100 frames 1000m cell size*