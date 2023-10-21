#include "spqlios/reim4/reim4_arena.h"

#include <gtest/gtest.h>

#include <cassert>

void init_random_values(uint64_t n, double* v) {
  for (uint64_t i = 0; i < n; ++i) v[i] = rand() - (RAND_MAX >> 1);
}

TEST(reim4_arena, reim4_extract_1col_from_reim_vector_same_result) {
  const uint64_t m = 65536;
  const uint64_t nrows = 64;
  const uint32_t col = rand() % (m >> 2);

  double* dst_ref = new double[nrows * 8];
  double* dst_sse = new double[nrows * 8];
  double* dst_avx2 = new double[nrows * 8];

  double** src = new double*[nrows];
  for (uint64_t i = 0; i < nrows; ++i) {
    src[i] = new double[m * 2];
    init_random_values(m * 2, src[i]);
  }

  reim4_extract_1col_from_reim_vector_ref(m, nrows, col, dst_ref, src);
  reim4_extract_1col_from_reim_vector_sse(m, nrows, col, dst_sse, src);
  reim4_extract_1col_from_reim_vector_avx2(m, nrows, col, dst_avx2, src);

  for (uint64_t i = 0; i < nrows * 8; ++i) {
    assert(dst_ref[i] == dst_sse[i]);
    assert(dst_ref[i] == dst_avx2[i]);
  }

  for (uint64_t i = 0; i < nrows; ++i) delete[] src[i];
  delete[] src;
  delete[] dst_avx2;
  delete[] dst_sse;
  delete[] dst_ref;
}
