#ifndef IDCT2_FFT2_CUH
#define IDCT2_FFT2_CUH

// Function declarations
__global__ void precomputeExpk_v2(cufftComplex *expkM, cufftComplex *expkN, cufftComplex *expkMN_1, cufftComplex *expkMN_2, const int M, const int N);
__global__ void precomputeExpk_v2(cufftDoubleComplex *expkM, cufftDoubleComplex *expkN, cufftDoubleComplex *expkMN_1, cufftDoubleComplex *expkMN_2, const int M, const int N);

template <typename T, typename TComplex>
__global__ void idct2d_preprocess(const T *input, TComplex *output, const int M, const int N,
                                  const int halfM, const int halfN,
                                  const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN,
                                  const TComplex *__restrict__ expkMN_1, const TComplex *__restrict__ expkMN_2);

template <typename T>
__global__ void idct2d_postprocess(const T *x, T *y, const int M, const int N, const int halfN);
template <typename T, typename TReal, typename TComplex>
void idct_2d_fft(const T *h_x, T *h_y, const int M, const int N);


#endif // IDCT2_FFT2_CUH