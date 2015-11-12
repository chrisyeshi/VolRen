#include "tfinteg2d.cuda.h"
#include <iostream>
#include "cutil_math.h"

__global__ void tfIntegrate2D_kernel(float* tf1d, float segLen, float* tf2dfull, float* tf2dback)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // integrate from y to x with y being the beginning of the ray segment and x being the end of the ray segment
    const float baseSample = 0.01;
    int dirSteps = x - y;
    int steps = (0 == dirSteps) ? 1 : abs(dirSteps);
    int dir = dirSteps / steps;
    float stepsize = segLen / float(steps);
    float4 full = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 back = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int step = 0; step < steps; ++step)
    {
        int tfIdx = y + step * dir;
        float4 spotColor, backColor;
        // sample
        spotColor.x = tf1d[4 * tfIdx + 0];
        spotColor.y = tf1d[4 * tfIdx + 1];
        spotColor.z = tf1d[4 * tfIdx + 2];
        spotColor.w = tf1d[4 * tfIdx + 3];
        // adjust
        spotColor.w = 1.f - pow(1.f - spotColor.w, stepsize / baseSample);
        spotColor.x *= spotColor.w;
        spotColor.y *= spotColor.w;
        spotColor.z *= spotColor.w;
        // weighted for front and back samples
        float weightBack = (steps - 1 == 0) ? 1.f : float(step) / float(steps - 1);
        backColor = spotColor * weightBack;
        // attenuate back
        back += backColor * (1.f - full.w);
        // attenuate full
        full += spotColor * (1.f - full.w);
    }
    // output
    int size = gridDim.x * blockDim.x;
    int idx = size * y + x;
    // full
    tf2dfull[4 * idx + 0] = full.x;
    tf2dfull[4 * idx + 1] = full.y;
    tf2dfull[4 * idx + 2] = full.z;
    tf2dfull[4 * idx + 3] = full.w;
    // back
    tf2dback[4 * idx + 0] = back.x;
    tf2dback[4 * idx + 1] = back.y;
    tf2dback[4 * idx + 2] = back.z;
    tf2dback[4 * idx + 3] = back.w;
}

void tfIntegrate2D(const float* tf1d, int size, float stepsize, float* tf2dfull, float* tf2dback)
{
    const int maxBlockSize = 16;
    dim3 dimBlock;
    dim3 dimGrid;
    if (size < maxBlockSize)
    {
        dimBlock = dim3(size, size);
        dimGrid = dim3(1, 1);
    } else
    {
        dimBlock = dim3(maxBlockSize, maxBlockSize);
        dimGrid = dim3(size / maxBlockSize, size / maxBlockSize);
    }

    float *devTF1d, *devTF2dFull, *devTF2dBack;
    const int size1d = 4 * size * sizeof(float);
    const int size2d = 4 * size * size * sizeof(float);
    cudaMalloc((void**)&devTF1d, size1d);
    cudaMalloc((void**)&devTF2dFull, size2d);
    cudaMalloc((void**)&devTF2dBack, size2d);
    cudaMemcpy(devTF1d, tf1d, size1d, cudaMemcpyHostToDevice);

    tfIntegrate2D_kernel<<<dimGrid, dimBlock>>>(devTF1d, stepsize, devTF2dFull, devTF2dBack);

    cudaMemcpy(tf2dfull, devTF2dFull, size2d, cudaMemcpyDeviceToHost);
    cudaMemcpy(tf2dback, devTF2dBack, size2d, cudaMemcpyDeviceToHost);
    cudaFree(devTF1d);
    cudaFree(devTF2dFull);
    cudaFree(devTF2dBack);
}
