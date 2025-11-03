#include <iostream>
#include <cuda_runtime.h>

__global__ void fillIndices(int *blockArray, int *threadArray, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < N) {
        blockArray[globalIdx] = blockIdx.x;
        threadArray[globalIdx] = threadIdx.x;
    }
}

int main() {
    const int N = 500;
    const int numBlocks = 16;
    const int threadsPerBlock = (N + numBlocks - 1) / numBlocks;

    int *d_blockArray, *d_threadArray;
    int *h_blockArray = new int[N];
    int *h_threadArray = new int[N];

    // Device-geheugen alloceren
    cudaMalloc(&d_blockArray, N * sizeof(int));
    cudaMalloc(&d_threadArray, N * sizeof(int));

    // Kernel uitvoeren
    fillIndices<<<numBlocks, threadsPerBlock>>>(d_blockArray, d_threadArray, N);
    cudaDeviceSynchronize();

    // Resultaten terughalen
    cudaMemcpy(h_blockArray, d_blockArray, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_threadArray, d_threadArray, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output tonen
    std::cout << "Index\tBlockIdx\tThreadIdx\n";
    std::cout << "----------------------------------\n";
    for (int i = 0; i < N; ++i) {
        std::cout << i << "\t" << h_blockArray[i] << "\t\t" << h_threadArray[i] << "\n";
    }

    // Opruimen
    cudaFree(d_blockArray);
    cudaFree(d_threadArray);
    delete[] h_blockArray;
    delete[] h_threadArray;

    return 0;
}
