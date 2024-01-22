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