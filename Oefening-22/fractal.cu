#include "image2d.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <iostream>

#define BLOCKSIZE 16

__global__ void CUDAKernel(int iterations, float xmin, float xmax, float ymin, float ymax, 
                           float *pOutput, int outputW, int outputH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= outputW || y >= outputH) return;

	float dx = (xmax - xmin) / (float)outputW;
	float dy = (ymax - ymin) / (float)outputH;

	float cx = xmin + x * dx;
	float cy = ymin + y * dy;

	// zx = Re(z), zy = Im(z)
	float zx = 0.0f;
	float zy = 0.0f;
	int iter = 0;

	while (zx*zx + zy*zy <= 4.0f && iter < iterations) {
		float xtemp = zx*zx - zy*zy + cx;
		zy = 2.0f*zx*zy + cy;
		zx = xtemp;
		iter++;
	}

	float value;
	if (iter == iterations) {
		value = 0.0f; // zwart
	} else {
		value = sqrt(iter / (float)iterations) * 255.0f;
	}
	pOutput[y * outputW + x] = value;
}

// If an error occurs, return false and set a description in 'errStr'
bool cudaFractal(int iterations, float xmin, float xmax, float ymin, float ymax, 
                 Image2D &output, std::string &errStr)
	{
		// We'll use an image of 512 pixels wide
		int ho = 512;
		int wo = ho * 3 / 2;
		output.resize(wo, ho);

		// And divide this in a number of blocks
		size_t xBlockSize = BLOCKSIZE;
		size_t yBlockSize = BLOCKSIZE;
		size_t numXBlocks = (wo/xBlockSize) + (((wo%xBlockSize) != 0)?1:0);
		size_t numYBlocks = (ho/yBlockSize) + (((ho%yBlockSize) != 0)?1:0);

		cudaError_t err;
		float *pDevOutput;

		err = cudaMalloc((void**)&pDevOutput, wo * ho * sizeof(float));
		if (err != cudaSuccess) {
			errStr = "Failed to allocate GPU memory";
			return false;
		}

		cudaEvent_t startEvt, stopEvt; // We'll use cuda events to time everything
		cudaEventCreate(&startEvt);
		cudaEventCreate(&stopEvt);

		cudaEventRecord(startEvt);

		dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
		dim3 gridDim(numXBlocks, numYBlocks);

		CUDAKernel<<<gridDim, blockDim>>>(iterations, xmin, xmax, ymin, ymax,
										  pDevOutput, wo, ho);

		cudaDeviceSynchronize(); // wacht tot kernel klaar is

		cudaEventRecord(stopEvt);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cout << "CUDA convolution kernel execution error code: " << err << std::endl;
		}

		err = cudaMemcpy(output.getBufferPointer(), pDevOutput, wo * ho * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			errStr = "Failed to copy data from GPU to host";
			cudaFree(pDevOutput);
			return false;
		}

		cudaFree(pDevOutput);

		float elapsed;
		cudaEventElapsedTime(&elapsed, startEvt, stopEvt);

		std::cout << "CUDA time elapsed: " << elapsed << " milliseconds" << std::endl;

		cudaEventDestroy(startEvt);
		cudaEventDestroy(stopEvt);

		return true;
	}
