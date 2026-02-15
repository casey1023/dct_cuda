#ifndef DCT2_FFT2_CUH
#define DCT2_FFT2_CUH

#include "../utils/cuda_utils.cuh"

#define TPB (16)

template <typename T>
__global__ void dct2d_preprocess(const T *x, T *y, const int M, const int N, const int halfN);

__global__ void precomputeExpk(cufftDoubleComplex *expkM, cufftDoubleComplex *expkN, const int M, const int N);
__global__ void precomputeExpk(cufftComplex *expkM, cufftComplex *expkN, const int M, const int N);

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void dct2d_postprocess(const TComplex *V, T *y, const int M, const int N,
                                                                  const int halfM, const int halfN, const T two_over_MN, const T four_over_MN,
                                                                  const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN);
template <typename T, typename TReal, typename TComplex>
void dct_2d_fft(const T *h_x, T *h_y, const int M, const int N);
#endif // DCT2_FFT2_CUH
