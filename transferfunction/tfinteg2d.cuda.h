#ifndef TFINTEG2D_CUDA_H
#define TFINTEG2D_CUDA_H

void tfIntegrate2D(const float* tf1d, int size, float segLen, float* tf2d, float *tf2dback);

#endif // TFINTEG2D_CUDA_H
