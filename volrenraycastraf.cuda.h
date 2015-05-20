#ifndef VOLRENRAYCASTRAF_CUDA_H
#define VOLRENRAYCASTRAF_CUDA_H

#include <cuda_runtime.h>

void rafcast(int volWidth, int volHeight, int volDepth, cudaArray* volArr,
        int tfSize, float stepSize, cudaTextureFilterMode filter, bool preinteg, cudaArray* tfArr,
        float scalarMin, float scalarMax,
        int texWidth, int texHeight, cudaArray* entryArr, cudaArray* exitArr,
        int layers, float* binDivs, float* rafPtr,
        float* mPtr, float* mvPtr, float near, float far, float* depPtr);

#endif // VOLRENRAYCASTRAF_CUDA_H
