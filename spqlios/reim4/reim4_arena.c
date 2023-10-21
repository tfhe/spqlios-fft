#include "reim4_arena.h"

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

// dst should contain: src[0](col),src[1](col),...,src[nrows-1](col)
// use scalar or sse code. Avx would be counterproductive a priori.
void reim4_extract_1col_from_reim_vector_ref(uint64_t m, uint64_t nrows, uint64_t col,
                                             double* dst,  // nrows * 8 doubles
                                             double** src  // a vector if nrows reim vectors
) {
  assert(col < (m >> 2));
  double* dst_ptr = dst;
  for (uint64_t i = 0; i < nrows; ++i, dst_ptr += 8) {
    const double* src_ptr = src[i] + (col << 2);
    dst_ptr[0] = src_ptr[0];
    dst_ptr[1] = src_ptr[1];
    dst_ptr[2] = src_ptr[2];
    dst_ptr[3] = src_ptr[3];

    src_ptr += m;
    dst_ptr[4] = src_ptr[0];
    dst_ptr[5] = src_ptr[1];
    dst_ptr[6] = src_ptr[2];
    dst_ptr[7] = src_ptr[3];
  }
}

void reim4_extract_1col_from_reim_vector_sse(uint64_t m, uint64_t nrows, uint64_t col,
                                             double* dst,  // nrows * 8 doubles
                                             double** src  // a vector if nrows reim vectors
) {
  assert(col < (m >> 2));

  double* dst_ptr = dst;
  for (uint64_t i = 0; i < nrows; ++i, dst_ptr += 8) {
    const double* src_ptr = src[i] + (col << 2);
    _mm_storeu_pd(dst_ptr, _mm_loadu_pd(src_ptr));
    _mm_storeu_pd(dst_ptr + 2, _mm_loadu_pd(src_ptr + 2));

    src_ptr += m;
    _mm_storeu_pd(dst_ptr + 4, _mm_loadu_pd(src_ptr));
    _mm_storeu_pd(dst_ptr + 6, _mm_loadu_pd(src_ptr + 2));
  }
}

void reim4_extract_1col_from_reim_vector_avx2(uint64_t m, uint64_t nrows, uint64_t col,
                                              double* dst,  // nrows * 8 doubles
                                              double** src  // a vector if nrows reim vectors
) {
  assert(col < (m >> 2));

  double* dst_ptr = dst;
  for (uint64_t i = 0; i < nrows; ++i, dst_ptr += 8) {
    const double* src_ptr = src[i] + (col << 2);
    _mm256_storeu_pd(dst_ptr, _mm256_loadu_pd(src_ptr));
    src_ptr += m;
    _mm256_storeu_pd(dst_ptr + 4, _mm256_loadu_pd(src_ptr));
  }
}
