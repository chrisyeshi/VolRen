#include "volrenraycastraf.cuda.h"
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "cutil_math.h"

static const int maxBlockSize2D = 16;
static texture<float, cudaTextureType3D, cudaReadModeElementType> volTex;
static texture<float4, cudaTextureType1D, cudaReadModeElementType> tfTex;
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

__host__ __device__ static int getBinID(float* binDivs, int layers, float scalar, int begLayer = 0)
{
    if (scalar == 1.f)
        return layers - 1;
    for (int iLayer = begLayer; iLayer < layers; ++iLayer)
    {
        float lower = binDivs[iLayer];
        float upper = binDivs[iLayer + 1];
        if (scalar >= lower && scalar < upper)
            return iLayer;
    }
    return -1;
}

// | 0  4  8 12 | | 0 |
// | 1  5  9 13 | | 1 |
// | 2  6 10 14 | | 2 |
// | 3  7 11 15 | | 3 |
__device__ float4 mat4x4_mult_vec4(float* mat, float4 vec)
{
    float4 out;
    out.x = mat[0] * vec.x + mat[4] * vec.y + mat[8] * vec.z + mat[12] * vec.w;
    out.y = mat[1] * vec.x + mat[5] * vec.y + mat[9] * vec.z + mat[13] * vec.w;
    out.z = mat[2] * vec.x + mat[6] * vec.y + mat[10] * vec.z + mat[14] * vec.w;
    out.w = mat[3] * vec.x + mat[7] * vec.y + mat[11] * vec.z + mat[15] * vec.w;
    return out;
}

__device__ float tf2sample(float tfVal)
{
    return tfVal + 0.5f;
}

__device__ float sample2tf(float sample)
{
    return sample - 0.5f;
}

__device__ float scalar2sample(float scalar, int tfSize)
{
    return scalar * float(tfSize);
}

__device__ float sample2scalar(float sample, int tfSize)
{
    return sample / float(tfSize);
}

__device__ float scalar2tf(float scalar, int tfSize)
{
    return sample2tf(scalar2sample(scalar, tfSize));
}

__device__ float tf2scalar(float tfVal, int tfSize)
{
    return sample2scalar(tf2sample(tfVal), tfSize);
}

__global__ static void castray(int volWidth, int volHeight, int volDepth,
                    int tfSize, float stepSize, bool preinteg,
                    float scalarMin, float scalarMax,
                    int texWidth, int texHeight, int layers, float* binDivs,
                    float* rafPtr,
                    float* mPtr, float* mvPtr, float near, float far, float* depPtr)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= texWidth || y >= texHeight)
        return;

    const float baseSample = 0.01f;
    float3 entry = make_float3(tex2D(entryTex, x + 0.5f, y + 0.5f));
    float3 exit = make_float3(tex2D(exitTex, x + 0.5f, y + 0.5f));
    float3 dir = normalize(exit - entry);
    float maxLength = length(exit - entry);
    float4 entryObj = mat4x4_mult_vec4(mPtr, make_float4(entry, 1.f));
    float4 entryView = mat4x4_mult_vec4(mvPtr, make_float4(entry, 1.f));
    float2 scalar = make_float2(0.f, 0.f);
    scalar.y = tex3D(volTex, entryObj.x, entryObj.y, entryObj.z);
    scalar.y = clamp(float((scalar.x - scalarMin) / (scalarMax - scalarMin)), 0.f, 1.f);
    float2 depth = make_float2(0.f, 0.f);
    depth.y = (-entryView.z - near) / (far - near);
    depth.y = clamp(depth.x, 0.f, 1.f);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int step = 1; step * stepSize < maxLength; ++step)
    {
        float3 spot = entry + dir * (step * stepSize);
        float4 spotObj = mat4x4_mult_vec4(mPtr, make_float4(spot, 1.f));
        float4 spotView = mat4x4_mult_vec4(mvPtr, make_float4(spot, 1.f));
        depth.x = (-spotView.z - near) / (far - near);
        depth.x = clamp(depth.x, 0.f, 1.f);
        scalar.x = tex3D(volTex, spotObj.x, spotObj.y, spotObj.z);
        scalar.x = clamp(float((scalar.x - scalarMin) / (scalarMax - scalarMin)), 0.f, 1.f);
        // if preintegration is not enabled
        if (!preinteg)
        {
            float4 spotColor = tex1D(tfTex, scalar2sample(scalar.x, tfSize));
            spotColor.w = 1.f - pow(1.f - spotColor.w, stepSize / baseSample);
            spotColor.x *= spotColor.w;
            spotColor.y *= spotColor.w;
            spotColor.z *= spotColor.w;
            float4 spotAtten = spotColor * (1.f - acc.w);
            int binID = getBinID(binDivs, layers, scalar.x);
            int outLoc = binID * texWidth * texHeight + y * texWidth + x;
            rafPtr[outLoc] += spotAtten.w;
            // accumulate
            acc += spotAtten;
            // depth
            if (depth.x < depPtr[outLoc])
                depPtr[outLoc] = depth.x;

        } else
        {
            float tfCoordBeg = scalar2tf(scalar.y, tfSize);
            float tfCoordEnd = scalar2tf(scalar.x, tfSize);
            // find the TF bucket
            int tfBeg = int(floor(tfCoordBeg));
            int tfEnd = int(floor(tfCoordEnd));
            // if they are in the same TF bucket
            if (tfBeg == tfEnd)
            {
                float4 spotColor = tex1D(tfTex, scalar2sample(scalar.x, tfSize));
                spotColor.w = 1.f - pow(1.f - spotColor.w, stepSize / baseSample);
                spotColor.x *= spotColor.w;
                spotColor.y *= spotColor.w;
                spotColor.z *= spotColor.w;
                float4 spotAtten = spotColor * (1.f - acc.w);
                int binID = getBinID(binDivs, layers, scalar.x);
                int outLoc = binID * texWidth * texHeight + y * texWidth + x;
                rafPtr[outLoc] += spotAtten.w;
                // accumulate
                acc += spotAtten;
                // depth
                if (depth.x < depPtr[outLoc])
                    depPtr[outLoc] = depth.x;

            } else
            {
                float4 spotColor, spotAtten;
                float miniStepsize, miniDepth;
                int binID, tfBin, outLoc;
                int dir = abs(tfEnd - tfBeg) / (tfEnd - tfBeg);

                tfBin = tfBeg + max(dir, 0);
                miniStepsize = (float(tfBin) - tfCoordBeg) / (tfCoordEnd - tfCoordBeg) * stepSize;
                spotColor = tex1D(tfTex, tf2sample(float(tfBin)));
                spotColor.w = 1.f - pow(1.f - spotColor.w, miniStepsize / baseSample);
                spotColor.x *= spotColor.w;
                spotColor.y *= spotColor.w;
                spotColor.z *= spotColor.w;
                spotAtten = spotColor * (1.f - acc.w);
                binID = getBinID(binDivs, layers, tf2scalar(float(tfBin), tfSize));
                outLoc = binID * texWidth * texHeight + y * texWidth + x;
                rafPtr[outLoc] += spotAtten.w;
                acc += spotAtten;
                miniDepth = (float(tfBin) - tfCoordBeg) / (tfCoordEnd - tfCoordBeg) * (depth.x - depth.y) + depth.y;
                if (miniDepth < depPtr[outLoc])
                    depPtr[outLoc] = miniDepth;

                for (int tfBin = tfBeg + max(dir, 0) + dir; tfBin != tfEnd + max(dir, 0); tfBin += dir)
                {
                    miniStepsize = float(dir) / (tfCoordEnd - tfCoordBeg) * stepSize;
                    spotColor = tex1D(tfTex, tf2sample(float(tfBin)));
                    spotColor.w = 1.f - pow(1.f - spotColor.w, miniStepsize / baseSample);
                    spotColor.x *= spotColor.w;
                    spotColor.y *= spotColor.w;
                    spotColor.z *= spotColor.w;
                    spotAtten = spotColor * (1.f - acc.w);
                    binID = getBinID(binDivs, layers, tf2scalar(float(tfBin), tfSize));
                    outLoc = binID * texWidth * texHeight + y * texWidth + x;
                    rafPtr[outLoc] += spotAtten.w;
                    acc += spotAtten;
                    miniDepth = (float(tfBin) - tfCoordBeg) / (tfCoordEnd - tfCoordBeg) * (depth.x - depth.y) + depth.y;
                    if (miniDepth < depPtr[outLoc])
                        depPtr[outLoc] = miniDepth;
                }

                miniStepsize = (tfCoordEnd - float(tfEnd + max(-dir, 0))) / (tfCoordEnd - tfCoordBeg) * stepSize;
                spotColor = tex1D(tfTex, tf2sample(tfCoordEnd));
                spotColor.w = 1.f - pow(1.f - spotColor.w, miniStepsize / baseSample);
                spotColor.x *= spotColor.w;
                spotColor.y *= spotColor.w;
                spotColor.z *= spotColor.w;
                spotAtten = spotColor * (1.f - acc.w);
                binID = getBinID(binDivs, layers, tf2scalar(tfCoordEnd, tfSize));
                outLoc = binID * texWidth * texHeight + y * texWidth + x;
                rafPtr[outLoc] += spotAtten.w;
                acc += spotAtten;
                if (depth.x < depPtr[outLoc])
                    depPtr[outLoc] = depth.x;
            }
        }
        if (acc.w > 0.999f)
            break;
        depth.y = depth.x;
        scalar.y = scalar.x;
    }
}

void rafcast(int volWidth, int volHeight, int volDepth, cudaArray *volArr,
             int tfSize, float stepSize, cudaTextureFilterMode filter, bool preinteg, cudaArray *tfArr,
             float scalarMin, float scalarMax,
             int texWidth, int texHeight, cudaArray *entryArr, cudaArray *exitArr,
             int layers, float *binDivs, float *rafPtr, float *mPtr, float *mvPtr, float near, float far, float *depPtr)
{
    // bind textures
    cudaBindTextureToArray(volTex, volArr);
    volTex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(tfTex, tfArr);
    tfTex.filterMode = filter;
    cudaBindTextureToArray(entryTex, entryArr);
    entryTex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(exitTex, exitArr);
    exitTex.filterMode = cudaFilterModeLinear;
    // clear raf and dep resource
    cudaMemset(rafPtr, 0, texWidth * texHeight * layers * sizeof(float));
    thrust::device_ptr<float> depPtrDev = thrust::device_pointer_cast(depPtr);
    thrust::fill(depPtrDev, depPtrDev + texWidth * texHeight * layers, 1.f);
    // RAF bin divisions
//    float* binDivs = new float [layers + 1];
//    fillBinDivs(binDivs, layers);
    float* devBinDivs;
    cudaMalloc(&devBinDivs, (layers + 1) * sizeof(float));
    cudaMemcpy(devBinDivs, binDivs, (layers + 1) * sizeof(float), cudaMemcpyHostToDevice);
    // cuda kernel
    dim3 dimBlock = getDimBlock2D(texWidth, texHeight);
    dim3 dimGrid = getDimGrid2D(texWidth, texHeight);
    castray<<<dimGrid, dimBlock>>>(volWidth, volHeight, volDepth,
                                   tfSize, stepSize, preinteg,
                                   scalarMin, scalarMax,
                                   texWidth, texHeight, layers, devBinDivs,
                                   rafPtr,
                                   mPtr, mvPtr, near, far, depPtr);
    // free memory
    cudaFree(devBinDivs);
//    delete [] binDivs;
    // unbind textures
    cudaUnbindTexture(exitTex);
    cudaUnbindTexture(entryTex);
    cudaUnbindTexture(tfTex);
    cudaUnbindTexture(volTex);
}
