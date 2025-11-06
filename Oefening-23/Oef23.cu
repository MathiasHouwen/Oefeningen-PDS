// reverse.cu
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(err));                                 \
      return EXIT_FAILURE;                                              \
    }                                                                   \
  } while (0)

const int N = 512;
const int BLOCKS = 16;
const int THREADS = 32;

__global__ void reverse_kernel(const int* in, int* out, int n) {
    // shared memory for this block (one element per thread)
    __shared__ int s[THREADS];

    int t = threadIdx.x;
    int b = blockIdx.x;
    int idx = b * blockDim.x + t; // global input index

    // safety: check bounds (shouldn't be needed for exact sizes but good practice)
    if (idx < n) {
        // lees relevant stuk in shared memory
        s[t] = in[idx];
    } else {
        s[t] = 0;
    }

    __syncthreads();

    // bereken de geschreven index: spiegel het gehele array
    // N-1 - idx is de correcte globale spiegel-index
    int write_idx = n - 1 - idx;

    if (idx < n) {
        // schrijf de waarde die in shared memory stond naar de spiegelpositie
        // we gebruiken s[t] (de waarde die deze thread in shared memory plaatste)
        out[write_idx] = s[t];
    }
}

int main() {
    int *h_in = new int[N];
    int *h_out = new int[N];

    for (int i = 0; i < N; ++i) h_in[i] = i;

    int *d_in = nullptr;
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel: 16 blocks, 32 threads
    reverse_kernel<<<BLOCKS, THREADS>>>(d_in, d_out, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // print input and output to stdout
    printf("Input  array (0..511):\n");
    for (int i = 0; i < N; ++i) {
        printf("%4d", h_in[i]);
        if ((i+1) % 16 == 0) printf("\n");
    }

    printf("\nOutput array (reversed):\n");
    for (int i = 0; i < N; ++i) {
        printf("%4d", h_out[i]);
        if ((i+1) % 16 == 0) printf("\n");
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;

    return 0;
}
