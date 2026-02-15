#ifndef IDCT_IDXST_CUH
#define IDCT_IDXST_CUH

#define TPB (16)

template <typename T, typename TComplex>
__global__ __launch_bounds__(TPB *TPB, 10) void idct_idxst_preprocess(const T *input, TComplex *output, const int M, const int N,
                                                                      const int halfM, const int halfN,
                                                                      const TComplex *__restrict__ expkM, const TComplex *__restrict__ expkN);
template <typename T>
__global__ void idct_idxst_postprocess(const T *x, T *y, const int M, const int N, const int halfN);

template <typename T, typename TReal, typename TComplex>
void idct_idxst(const T *h_x, T *h_y, const int M, const int N);
#endif // IDCT_IDXST_CUH