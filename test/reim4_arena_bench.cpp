#include <benchmark/benchmark.h>

#include "spqlios/reim4/reim4_arena.h"

void init_random_values(uint64_t n, double* v) {
  for (uint64_t i = 0; i < n; ++i) v[i] = rand() - (RAND_MAX >> 1);
}

void reim4_extract_1col_from_reim_vector_setup(const uint64_t m, const uint64_t nrows, double*& dst, double**& src) {
  dst = new double[nrows * 8];

  src = new double*[nrows];
  for (uint64_t i = 0; i < nrows; ++i) {
    src[i] = new double[m * 2];
    init_random_values(m * 2, src[i]);
  }
}

void reim4_extract_1col_from_reim_vector_teardown(const uint64_t m, const uint64_t nrows, double*& dst, double**& src) {
  for (uint64_t i = 0; i < nrows; ++i) delete[] src[i];
  delete[] src;
  delete[] dst;
}

template <void (*fnc)(uint64_t m, uint64_t nrows, uint64_t col, double* dst, double** src)>
void benchmark_reim4_extract_1col_from_reim_vector(benchmark::State& state) {
  const uint64_t m = state.range(0);
  const uint64_t nrows = state.range(1);
  const uint32_t col = rand() % (m >> 2);

  double** src;
  double* dst;
  reim4_extract_1col_from_reim_vector_setup(m, nrows, dst, src);

  for (auto _ : state) {
    fnc(m, nrows, col, dst, src);
  }

  reim4_extract_1col_from_reim_vector_teardown(m, nrows, dst, src);
}

#define ARGS ArgsProduct({{1024, 8192, 32768, 65536}, {1, 16, 32, 64}})

BENCHMARK(benchmark_reim4_extract_1col_from_reim_vector<reim4_extract_1col_from_reim_vector_ref>)->ARGS;
BENCHMARK(benchmark_reim4_extract_1col_from_reim_vector<reim4_extract_1col_from_reim_vector_sse>)->ARGS;
BENCHMARK(benchmark_reim4_extract_1col_from_reim_vector<reim4_extract_1col_from_reim_vector_avx2>)->ARGS;

// Run the benchmark
BENCHMARK_MAIN();
