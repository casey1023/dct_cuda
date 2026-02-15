#ifndef IDXST_IDCT_CUH
#define IDXST_IDCT_CUH

#define TPB (16)

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idxst_idct_preprocess(const T *input, TComplex *output, const int M, const int N,
                                                                      const int halfM, const int halfN,
                                                                      const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN);
template <typename T>
__global__ void idxst_idct_postprocess(const T *x, T *y, const int M, const int N, const int halfN);

template <typename T, typename TReal, typename TComplex>
void idxst_idct(const T *h_x, T *h_y, const int M, const int N);

#endif // IDXST_IDCT_CUH