#include "volrenraycastcuda.cuda.h"
#include <stdio.h>
#include "cutil_math.h"

__constant__ int volWidth, volHeight, volDepth;
__constant__ int nLights;
__constant__ CudaLight lights[10];

static const int maxBlockSize2D = 16;
static texture<float, cudaTextureType3D, cudaReadModeElementType> volTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> tfFullTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> tfBackTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> entryTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> exitTex;

#define cc(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

__device__ static float3 makeGradient(float3 spot)
{
    float3 gradient;
    gradient.x = 0.5 * (tex3D(volTex, spot.x * volWidth + 1.f, spot.y * volHeight, spot.z * volDepth)
                      - tex3D(volTex, spot.x * volWidth - 1.f, spot.y * volHeight, spot.z * volDepth));
    gradient.y = 0.5 * (tex3D(volTex, spot.x * volWidth, spot.y * volHeight + 1.f, spot.z * volDepth)
                      - tex3D(volTex, spot.x * volWidth, spot.y * volHeight - 1.f, spot.z * volDepth));
    gradient.z = 0.5 * (tex3D(volTex, spot.x * volWidth, spot.y * volHeight, spot.z * volDepth + 1.f)
                      - tex3D(volTex, spot.x * volWidth, spot.y * volHeight, spot.z * volDepth - 1.f));
    return gradient;
}

__device__ static float3 entryGradient(float2 viewSpot)
{
    float delta = 0.1f;
    float3 left = make_float3(tex2D(entryTex, viewSpot.x - delta, viewSpot.y));
    float3 right = make_float3(tex2D(entryTex, viewSpot.x + delta, viewSpot.y));
    float3 top = make_float3(tex2D(entryTex, viewSpot.x, viewSpot.y + delta));
    float3 bottom = make_float3(tex2D(entryTex, viewSpot.x, viewSpot.y - delta));
    return cross(top - bottom, right - left);
}

__device__ static float4 getLightFactor(float3 grad, float3 view)
{
    if (nLights == 0)
        return make_float4(1.f, 1.f, 1.f, 1.f);
    float3 V = normalize(-view);
    float3 N = normalize(-grad);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = 0; i < nLights; ++i)
    {
        float3 kd = lights[i].diffuse;
        float3 ka = lights[i].ambient;
        float3 ks = lights[i].specular;
        float shininess = lights[i].shininess;
        float3 L = normalize(make_float3(0.f, 0.f, 0.f) - lights[i].direction);
        float3 R = normalize(make_float3(0.f, 0.f, 0.f) - reflect(L, N));
        float3 diffuse = kd * max(dot(L, N), 0.f);
        float3 specular = ks * pow(max(dot(R, V), 0.f), shininess);
        float3 cf = ka + diffuse + specular;
        float af = 1.f;
        acc += make_float4(cf.x, cf.y, cf.z, af);
    }
    return make_float4(acc.x, acc.y, acc.z, 1.f);
}

__global__ static void castray(int tfWidth, int tfHeight, float stepSize,
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
    scalar.y = tex3D(volTex, entry.x * volWidth, entry.y * volHeight, entry.z * volDepth);
    scalar.y = clamp(float((scalar.y - scalarMin) / (scalarMax - scalarMin)), 0.f, 1.f);
    float3 spotCurr;
    float4 lfPrev = getLightFactor(entryGradient(make_float2(x + 0.5f, y + 0.5f)), dir);
    float4 lfCurr;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int step = 1; step * stepSize < maxLength; ++step)
    {
        spotCurr = entry + dir * (step * stepSize);
        scalar.x = tex3D(volTex, spotCurr.x * volWidth, spotCurr.y * volHeight, spotCurr.z * volDepth);
        scalar.x = clamp(float((scalar.x - scalarMin) / (scalarMax - scalarMin)), 0.f, 1.f);
        float4 colorFull = tex2D(tfFullTex, scalar.x * tfWidth, scalar.y * tfHeight);
        float4 colorBack = tex2D(tfBackTex, scalar.x * tfWidth, scalar.y * tfHeight);
        float4 colorFront = colorFull - colorBack;
        lfCurr = getLightFactor(makeGradient(spotCurr), dir);
        acc += (colorBack * lfCurr + colorFront * lfPrev) * (1.0 - acc.w);
        if (acc.w > 0.999f)
            break;
        scalar.y = scalar.x;
        lfPrev = lfCurr;
    }
    outPtr[3 * (texWidth * y + x) + 0] = acc.x;
    outPtr[3 * (texWidth * y + x) + 1] = acc.y;
    outPtr[3 * (texWidth * y + x) + 2] = acc.z;
}

void cudacast(int ivolWidth, int ivolHeight, int ivolDepth, cudaArray* volArr,
              int tfWidth, int tfHeight, float stepSize, cudaTextureFilterMode filter, cudaArray* tfFullArr, cudaArray *tfBackArr,
              float scalarMin, float scalarMax,
              int texWidth, int texHeight, cudaArray *entryArr, cudaArray *exitArr, float *outPtr)
{
    cudaMemcpyToSymbol(volWidth, &ivolWidth, sizeof(int));
    cudaMemcpyToSymbol(volHeight, &ivolHeight, sizeof(int));
    cudaMemcpyToSymbol(volDepth, &ivolDepth, sizeof(int));

    cudaBindTextureToArray(volTex, volArr);
    volTex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(tfFullTex, tfFullArr);
    tfFullTex.filterMode = filter;
    cudaBindTextureToArray(tfBackTex, tfBackArr);
    tfBackTex.filterMode = filter;
    cudaBindTextureToArray(entryTex, entryArr);
    entryTex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(exitTex, exitArr);
    exitTex.filterMode = cudaFilterModeLinear;

    dim3 dimBlock = getDimBlock2D(texWidth, texHeight);
    dim3 dimGrid = getDimGrid2D(texWidth, texHeight);
    castray<<<dimGrid, dimBlock>>>(tfWidth, tfHeight, stepSize,
                                   scalarMin, scalarMax,
                                   texWidth, texHeight, outPtr);

    cudaUnbindTexture(exitTex);
    cudaUnbindTexture(entryTex);
    cudaUnbindTexture(tfFullTex);
    cudaUnbindTexture(tfBackTex);
    cudaUnbindTexture(volTex);
}

void cudaSetLights(int inLights, CudaLight ilights[])
{
    cc(cudaMemcpyToSymbol(nLights, &inLights, sizeof(int)));
    cc(cudaMemcpyToSymbol(lights, &ilights[0], 10 * sizeof(CudaLight)));
}
