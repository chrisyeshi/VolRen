#ifndef VOLRENRAYCASTCUDA_CUDA_H
#define VOLRENRAYCASTCUDA_CUDA_H

#include <cuda_runtime.h>

void cudacast(
        int volWidth, int volHeight, int volDepth, cudaArray* volArr,
        int tfWidth, int tfHeight, float stepSize, cudaTextureFilterMode filter, cudaArray* tfArr,
        float scalarMin, float scalarMax,
        int texWidth, int texHeight, cudaArray* entryArr, cudaArray* exitArr, float* outPtr);

#endif // VOLRENRAYCASTCUDA_CUDA_H
