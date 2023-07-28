#ifndef SPQLIOS_REIM_FFT_H
#define SPQLIOS_REIM_FFT_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "reim_fft_public.h"

EXPORT void reim_fftvec_addmul_fma(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                   const double* b);
EXPORT void reim_fftvec_addmul_ref(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                   const double* b);

EXPORT void reim_fftvec_mul_fma(const REIM_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b);
EXPORT void reim_fftvec_mul_ref(const REIM_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b);


/** @brief r = x from ZnX (coeffs as signed int32_t's ) to double */
EXPORT void reim_from_znx32_ref(const REIM_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x);
EXPORT void reim_from_znx32_avx2_fma(const REIM_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x);

/**
 * @brief compute the fft evaluations of P in place
 * fft(data) = fft_rec(data, i);
 * function fft_rec(data, omega) {
 *   if #data = 1: return data
 *   let s = sqrt(omega) w. re(s)>0
 *   let (u,v) = merge_fft(data, s)
 *   return [fft_rec(u, s), fft_rec(v, -s)]
 * }
 * @param tables precomputed tables (contains all the powers of omega in the order they are used)
 * @param data vector of m complexes (coeffs as input, evals as output)
 */
EXPORT void reim_fft_ref(const REIM_FFT_PRECOMP* tables, double* data);
EXPORT void reim_fft_avx2_fma(const REIM_FFT_PRECOMP* tables, double* data);

/**
 * @brief compute the ifft evaluations of P in place
 * ifft(data) = ifft_rec(data, i);
 * function ifft_rec(data, omega) {
 *   if #data = 1: return data
 *   let s = sqrt(omega) w. re(s)>0
 *   let (u,v) = data
 *   return split_fft([ifft_rec(u, s), ifft_rec(v, -s)],s)
 * }
 * @param itables precomputed tables (contains all the powers of omega in the order they are used)
 * @param data vector of m complexes (coeffs as input, evals as output)
 */
EXPORT void reim_ifft_ref(const REIM_IFFT_PRECOMP* itables, double* data);
EXPORT void reim_ifft_avx2_fma(const REIM_IFFT_PRECOMP* itables, double* data);

/// new compressed implementation

/** @brief naive FFT code mod X^m-exp(2i.pi.entry_pwr) */
EXPORT void reim_naive_fft(uint64_t m, double entry_pwr, double* re, double* im);

/** @brief 16-dimensional FFT with precomputed omegas */
EXPORT void reim_fft16_avx_fma(double* dre, double* dim, const void* omega);
EXPORT void reim_fft16_ref(double* dre, double* dim, const void* omega);

/** @brief precompute omegas so that reim_fft16 functions */
EXPORT void fill_reim_fft16_omegas(const double entry_pwr, double** omg);

#endif  // SPQLIOS_REIM_FFT_H
