#ifndef SPQLIOS_REIM4_ARENA_H
#define SPQLIOS_REIM4_ARENA_H

#include <stdint.h>

#include "../commons.h"

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

#endif  // SPQLIOS_REIM4_ARENA_H
