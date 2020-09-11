// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
// 
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <stdio.h>
#include <vector_types.h>
#include <time.h>

#include "PPM.hh"

using namespace std;
using namespace ppm;

#define MAX_THREADS 1024



//-------------------------------------------------------------------------------

// specify the gamma value to be applied
__device__ __constant__ float gpuGamma[1];

__device__ float applyGamma(const float& _src, const float _gamma)
{
	return 255.0f * __powf(_src / 255.0f, _gamma);
}

/* compute gamma correction on the float image _src of resolution dim,
 outputs the gamma corrected image should be stored in_dst[blockIdx.x *
 blockDim.x + threadIdx.x]. Each thread computes on pixel element.
 */__global__ void gammaKernel(float *_dst, const float* _src1, const float* _src2, int _w)
{
	int x = blockIdx.x * MAX_THREADS + threadIdx.x;
	int y = blockIdx.y;
	int pos = y * _w + x;

	if (x < _w)
	{
		_dst[pos] = abs(_src1[pos] - _src2[pos]);
	}
}

//-------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	int acount = 1; // parse command line

	if (argc < 4)
	{
		printf("usage: %s <inImg1> <inImg2> <outImg>\n", argv[0]);
		exit(1);
	}

	float* img1;
	float* img2;

	clock_t start, end;
     	double cpu_time_used;
     
     	
	int w, h;
	readPPM(argv[acount++], w, h, &img1);
	readPPM(argv[acount++], w, h, &img2);

	//float gamma = atof(argv[acount++]);

	int nPix = w * h;

	float* gpuImg1;
	float* gpuImg2;
	float* gpuResImg;

	//-------------------------------------------------------------------------------
	printf("Executing the GPU Version\n");
	// copy the image to the device
	cudaMalloc((void**) &gpuImg1, nPix * 3 * sizeof(float));
	cudaMalloc((void**) &gpuImg2, nPix * 3 * sizeof(float));
	cudaMalloc((void**) &gpuResImg, nPix * 3 * sizeof(float));
	cudaMemcpy(gpuImg1, img1, nPix * 3 * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(gpuImg2, img2, nPix * 3 * sizeof(float),
			cudaMemcpyHostToDevice);

	
	// calculate the block dimensions
	dim3 threadBlock(MAX_THREADS);
	// select the number of blocks vertically (*3 because of RGB)
	dim3 blockGrid((w * 3) / MAX_THREADS + 1, h, 1);
	printf("bl/thr: %d  %d %d\n", blockGrid.x, blockGrid.y, threadBlock.x);
	
	int i;
	for (i = 1; i <= 10; i++) {
		
		start = clock();
		gammaKernel<<< blockGrid, threadBlock >>>(gpuResImg, gpuImg1, gpuImg2, w * 3);
		cudaDeviceSynchronize();
		end = clock();
     		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%d maxthreads - iteration %d is: %lf\n", MAX_THREADS, i, cpu_time_used);

	}
	// download result
	cudaMemcpy(img1, gpuResImg, nPix * 3 * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(gpuResImg);
	cudaFree(gpuImg1);
	cudaFree(gpuImg2);

	writePPM(argv[acount++], w, h, (float*) img1);

	delete[] img1;
	delete[] img2;

	printf("  done\n");
}

