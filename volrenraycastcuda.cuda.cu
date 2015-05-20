#include "volrenraycastcuda.cuda.h"
#include <stdio.h>
#include "cutil_math.h"

static const int maxBlockSize2D = 16;
static texture<float, cudaTextureType3D, cudaReadModeElementType> volTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> tfTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> entryTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> exitTex;

static dim3 getDimBlock2D(int w, int h)
{
    dim3 dimBlock;
    if (w < maxBlockSize2D)
        dimBlock.x = w;
    else
        dimBlock.x = maxBlockSize2D;
    if (h < maxBlockSize2D)
        dimBlock.y = h;
    else
        dimBlock.y = maxBlockSize2D;
    return dimBlock;
}

static dim3 getDimGrid2D(int w, int h)
{
    dim3 dimGrid;
    if (w < maxBlockSize2D)
        dimGrid.x = 1;
    else
        dimGrid.x = int(ceil(float(w) / maxBlockSize2D));
    if (h < maxBlockSize2D)
        dimGrid.y = 1;
    else
        dimGrid.y = int(ceil(float(h) / maxBlockSize2D));
    return dimGrid;
}

__global__ static void castray(int volWidth, int volHeight, int volDepth,
                        int tfWidth, int tfHeight, float stepSize,
                        float scalarMin, float scalarMax,
                        int texWidth, int texHeight, float* outPtr)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= texWidth || y >= texHeight)
        return;

    float3 entry = make_float3(tex2D(entryTex, x + 0.5f, y + 0.5f));
    float3 exit = make_float3(tex2D(exitTex, x + 0.5f, y + 0.5f));
    float3 dir = normalize(exit - entry);
    float maxLength = length(exit - entry);
    float2 scalar = make_float2(0.f, 0.f);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    int step = 0;
    for (; step * stepSize < maxLength; ++step)
    {
        float3 spot = entry + dir * (step * stepSize);
        scalar.x = tex3D(volTex, spot.x * volWidth, spot.y * volHeight, spot.z * volDepth);
        scalar.x = clamp(float((scalar.x - scalarMin) / (scalarMax - scalarMin)), 0.f, 1.f);
        float4 spotColor = tex2D(tfTex, scalar.x * tfWidth, scalar.y * tfHeight);
        acc += spotColor * (1.f - acc.w);
        if (acc.w > 0.999f)
            break;
        scalar.y = scalar.x;
    }
    outPtr[3 * (texWidth * y + x) + 0] = acc.x;
    outPtr[3 * (texWidth * y + x) + 1] = acc.y;
    outPtr[3 * (texWidth * y + x) + 2] = acc.z;
}

void cudacast(int volWidth, int volHeight, int volDepth, cudaArray* volArr,
              int tfWidth, int tfHeight, float stepSize, cudaTextureFilterMode filter, cudaArray* tfArr,
              float scalarMin, float scalarMax,
              int texWidth, int texHeight, cudaArray *entryArr, cudaArray *exitArr, float *outPtr)
{
    cudaBindTextureToArray(volTex, volArr);
    volTex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(tfTex, tfArr);
    tfTex.filterMode = filter;
    cudaBindTextureToArray(entryTex, entryArr);
    entryTex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(exitTex, exitArr);
    exitTex.filterMode = cudaFilterModeLinear;

    dim3 dimBlock = getDimBlock2D(texWidth, texHeight);
    dim3 dimGrid = getDimGrid2D(texWidth, texHeight);
    castray<<<dimGrid, dimBlock>>>(volWidth, volHeight, volDepth,
                                   tfWidth, tfHeight, stepSize,
                                   scalarMin, scalarMax,
                                   texWidth, texHeight, outPtr);

    cudaUnbindTexture(exitTex);
    cudaUnbindTexture(entryTex);
    cudaUnbindTexture(tfTex);
    cudaUnbindTexture(volTex);
}
