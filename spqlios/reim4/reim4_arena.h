#ifndef SPQLIOS_REIM4_ARENA_H
#define SPQLIOS_REIM4_ARENA_H

#include <stdint.h>

#include "../commons.h"

// dst should contain: src[0](col),src[1](col),...,src[nrows-1](col)
// use scalar or sse code. Avx would be counterproductive a priori.
EXPORT void reim4_extract_1col_from_reim_vector_ref(uint64_t m, uint64_t nrows, uint64_t col,
                                                    double* dst,  // nrows * 8 doubles
                                                    double** src  // a vector if nrows reim vectors
);

EXPORT void reim4_extract_1col_from_reim_vector_sse(uint64_t m, uint64_t nrows, uint64_t col,
                                                    double* dst,  // nrows * 8 doubles
                                                    double** src  // a vector if nrows reim vectors
);

EXPORT void reim4_extract_1col_from_reim_vector_avx2(uint64_t m, uint64_t nrows, uint64_t col,
                                                     double* dst,  // nrows * 8 doubles
                                                     double** src  // a vector if nrows reim vectors
);

// dest should contain:
// src[0](col),src[0](col+1),
// src[1](col),src[1](col+1),
// src[nrows-1](col),src[nrows-1](col+1)
// use just scalar code a priori (does not need to be efficient)
EXPORT void reim4_extract_2cols_from_reim_vector_ref(uint64_t m, uint64_t nrows, uint64_t col,
                                                     double* dst,  // nrows * 16 doubles
                                                     double** src  // a vector if nrows reim vectors
);

#endif  // SPQLIOS_REIM4_ARENA_H
