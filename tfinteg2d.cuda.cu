#include "tfinteg2d.cuda.h"
#include <iostream>

__global__ void tfIntegrate2D_kernel(float* tf1d, float segLen, float* tf2d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // integrate from y to x with y being the beginning of the ray segment and x being the end of the ray segment
    const float baseSample = 0.01;
    int dirSteps = x - y;
    int steps = (0 == dirSteps) ? 1 : abs(dirSteps);
    int dir = dirSteps / steps;
    float stepsize = segLen / float(steps);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int step = 0; step < steps; ++step)
    {
        int tfIdx = y + step * dir;
        float4 spotColor;
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
        // attenuate
        acc.x += spotColor.x * (1.f - acc.w);
        acc.y += spotColor.y * (1.f - acc.w);
        acc.z += spotColor.z * (1.f - acc.w);
        acc.w += spotColor.w * (1.f - acc.w);
    }
    // output
    int size = gridDim.x * blockDim.x;
    int idx = size * y + x;
    tf2d[4 * idx + 0] = acc.x;
    tf2d[4 * idx + 1] = acc.y;
    tf2d[4 * idx + 2] = acc.z;
    tf2d[4 * idx + 3] = acc.w;
}

void tfIntegrate2D(const float* tf1d, int size, float stepsize, float* tf2d)
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

    float *devTF1d, *devTF2d;
    const int size1d = 4 * size * sizeof(float);
    const int size2d = 4 * size * size * sizeof(float);
    cudaMalloc((void**)&devTF1d, size1d);
    cudaMalloc((void**)&devTF2d, size2d);
    cudaMemcpy(devTF1d, tf1d, size1d, cudaMemcpyHostToDevice);

    tfIntegrate2D_kernel<<<dimGrid, dimBlock>>>(devTF1d, stepsize, devTF2d);

    cudaMemcpy(tf2d, devTF2d, size2d, cudaMemcpyDeviceToHost);
    cudaFree(devTF1d);
    cudaFree(devTF2d);
}
