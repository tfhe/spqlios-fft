# shifted FFT over X^16-i
# 1st argument (rdi) contains 16 complexes
# 2nd argument (rsi) contains: 8 complexes
#     omega,alpha,beta,j.beta,gamma,j.gamma,k.gamma,kj.gamma
#     alpha = sqrt(omega), beta = sqrt(alpha), gamma = sqrt(beta)
#     j = sqrt(i), k=sqrt(j)
.globl cplx_fft16_avx_fma
cplx_fft16_avx_fma:
vmovupd (%rdi),%ymm8
vmovupd 0x20(%rdi),%ymm9
vmovupd 0x40(%rdi),%ymm10
vmovupd 0x60(%rdi),%ymm11
vmovupd 0x80(%rdi),%ymm12
vmovupd 0xa0(%rdi),%ymm13
vmovupd 0xc0(%rdi),%ymm14
vmovupd 0xe0(%rdi),%ymm15

.first_pass:
vmovupd (%rsi),%xmm0                  /* omri   */
vinsertf128 $1, %xmm0, %ymm0, %ymm0   /* omriri */
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: omiiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: omrrrr */
vshufpd $5, %ymm12, %ymm12, %ymm4
vshufpd $5, %ymm13, %ymm13, %ymm5
vshufpd $5, %ymm14, %ymm14, %ymm6
vshufpd $5, %ymm15, %ymm15, %ymm7
vmulpd %ymm4,%ymm1,%ymm4
vmulpd %ymm5,%ymm1,%ymm5
vmulpd %ymm6,%ymm1,%ymm6
vmulpd %ymm7,%ymm1,%ymm7
vfmaddsub231pd  %ymm12, %ymm0, %ymm4     # ymm4 = (ymm0 * ymm12) +/- ymm4
vfmaddsub231pd  %ymm13, %ymm0, %ymm5
vfmaddsub231pd  %ymm14, %ymm0, %ymm6
vfmaddsub231pd  %ymm15, %ymm0, %ymm7
vsubpd %ymm4,%ymm8,%ymm12
vsubpd %ymm5,%ymm9,%ymm13
vsubpd %ymm6,%ymm10,%ymm14
vsubpd %ymm7,%ymm11,%ymm15
vaddpd %ymm4,%ymm8,%ymm8
vaddpd %ymm5,%ymm9,%ymm9
vaddpd %ymm6,%ymm10,%ymm10
vaddpd %ymm7,%ymm11,%ymm11

.second_pass:
vmovupd 16(%rsi),%xmm0                /* omri   */
vinsertf128 $1, %xmm0, %ymm0, %ymm0   /* omriri */
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: omiiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: omrrrr */
vshufpd $5, %ymm10, %ymm10, %ymm4
vshufpd $5, %ymm11, %ymm11, %ymm5
vshufpd $5, %ymm14, %ymm14, %ymm6
vshufpd $5, %ymm15, %ymm15, %ymm7
vmulpd %ymm4,%ymm1,%ymm4
vmulpd %ymm5,%ymm1,%ymm5
vmulpd %ymm6,%ymm0,%ymm6
vmulpd %ymm7,%ymm0,%ymm7
vfmaddsub231pd  %ymm10, %ymm0, %ymm4     # ymm4 = (ymm0 * ymm10) +/- ymm4
vfmaddsub231pd  %ymm11, %ymm0, %ymm5
vfmsubadd231pd  %ymm14, %ymm1, %ymm6
vfmsubadd231pd  %ymm15, %ymm1, %ymm7
vsubpd %ymm4,%ymm8,%ymm10
vsubpd %ymm5,%ymm9,%ymm11
vaddpd %ymm6,%ymm12,%ymm14
vaddpd %ymm7,%ymm13,%ymm15
vaddpd %ymm4,%ymm8,%ymm8
vaddpd %ymm5,%ymm9,%ymm9
vsubpd %ymm6,%ymm12,%ymm12
vsubpd %ymm7,%ymm13,%ymm13

.third_pass:
vmovupd 32(%rsi),%xmm0                /* gamma   */
vmovupd 48(%rsi),%xmm2                /* delta   */
vinsertf128 $1, %xmm0, %ymm0, %ymm0
vinsertf128 $1, %xmm2, %ymm2, %ymm2
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: gama.iiii */
vshufpd $15, %ymm2, %ymm2, %ymm3      /* ymm3: delta.iiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: gama.rrrr */
vshufpd $0,  %ymm2, %ymm2, %ymm2      /* ymm2: delta.rrrr */
vshufpd $5, %ymm9, %ymm9, %ymm4
vshufpd $5, %ymm11, %ymm11, %ymm5
vshufpd $5, %ymm13, %ymm13, %ymm6
vshufpd $5, %ymm15, %ymm15, %ymm7
vmulpd %ymm4,%ymm1,%ymm4
vmulpd %ymm5,%ymm0,%ymm5
vmulpd %ymm6,%ymm3,%ymm6
vmulpd %ymm7,%ymm2,%ymm7
vfmaddsub231pd  %ymm9, %ymm0, %ymm4     # ymm4 = (ymm0 * ymm10) +/- ymm4
vfmsubadd231pd  %ymm11, %ymm1, %ymm5
vfmaddsub231pd  %ymm13, %ymm2, %ymm6
vfmsubadd231pd  %ymm15, %ymm3, %ymm7
vsubpd %ymm4,%ymm8,%ymm9
vaddpd %ymm5,%ymm10,%ymm11
vsubpd %ymm6,%ymm12,%ymm13
vaddpd %ymm7,%ymm14,%ymm15
vaddpd %ymm4,%ymm8,%ymm8
vsubpd %ymm5,%ymm10,%ymm10
vaddpd %ymm6,%ymm12,%ymm12
vsubpd %ymm7,%ymm14,%ymm14

.fourth_pass:
vmovupd 64(%rsi),%ymm0                /* gamma   */
vmovupd 96(%rsi),%ymm2                /* delta   */
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: gama.iiii */
vshufpd $15, %ymm2, %ymm2, %ymm3      /* ymm3: delta.iiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: gama.rrrr */
vshufpd $0,  %ymm2, %ymm2, %ymm2      /* ymm2: delta.rrrr */
vperm2f128 $0x31,%ymm10,%ymm8,%ymm4   # ymm4 contains c1,c5 -- x gamma
vperm2f128 $0x31,%ymm11,%ymm9,%ymm5   # ymm5 contains c3,c7 -- x igamma
vperm2f128 $0x31,%ymm14,%ymm12,%ymm6  # ymm6 contains c9,c13 -- x delta
vperm2f128 $0x31,%ymm15,%ymm13,%ymm7  # ymm7 contains c11,c15 -- x idelta
vperm2f128 $0x20,%ymm10,%ymm8,%ymm8   # ymm8 contains c0,c4
vperm2f128 $0x20,%ymm11,%ymm9,%ymm9   # ymm9 contains c2,c6
vperm2f128 $0x20,%ymm14,%ymm12,%ymm10 # ymm10 contains c8,c12
vperm2f128 $0x20,%ymm15,%ymm13,%ymm11 # ymm11 contains c10,c14
vshufpd $5, %ymm4, %ymm4, %ymm12
vshufpd $5, %ymm5, %ymm5, %ymm13
vshufpd $5, %ymm6, %ymm6, %ymm14
vshufpd $5, %ymm7, %ymm7, %ymm15
vmulpd %ymm12,%ymm1,%ymm12
vmulpd %ymm13,%ymm0,%ymm13
vmulpd %ymm14,%ymm3,%ymm14
vmulpd %ymm15,%ymm2,%ymm15
vfmaddsub231pd  %ymm4, %ymm0, %ymm12     # ymm12 = (ymm0 * ymm4) +/- ymm12
vfmsubadd231pd  %ymm5, %ymm1, %ymm13
vfmaddsub231pd  %ymm6, %ymm2, %ymm14
vfmsubadd231pd  %ymm7, %ymm3, %ymm15
vsubpd %ymm12,%ymm8,%ymm4
vaddpd %ymm13,%ymm9,%ymm5
vsubpd %ymm14,%ymm10,%ymm6
vaddpd %ymm15,%ymm11,%ymm7
vaddpd %ymm12,%ymm8,%ymm8
vsubpd %ymm13,%ymm9,%ymm9
vaddpd %ymm14,%ymm10,%ymm10
vsubpd %ymm15,%ymm11,%ymm11

vperm2f128 $0x20,%ymm6,%ymm10,%ymm12  # ymm4 contains c1,c5 -- x gamma
vperm2f128 $0x20,%ymm7,%ymm11,%ymm13  # ymm5 contains c3,c7 -- x igamma
vperm2f128 $0x31,%ymm6,%ymm10,%ymm14  # ymm6 contains c9,c13 -- x delta
vperm2f128 $0x31,%ymm7,%ymm11,%ymm15  # ymm7 contains c11,c15 -- x idelta
vperm2f128 $0x31,%ymm4,%ymm8,%ymm10   # ymm10 contains c8,c12
vperm2f128 $0x31,%ymm5,%ymm9,%ymm11   # ymm11 contains c10,c14
vperm2f128 $0x20,%ymm4,%ymm8,%ymm8    # ymm8 contains c0,c4
vperm2f128 $0x20,%ymm5,%ymm9,%ymm9    # ymm9 contains c2,c6

.save_and_return:
vmovupd %ymm8,(%rdi)
vmovupd %ymm9,0x20(%rdi)
vmovupd %ymm10,0x40(%rdi)
vmovupd %ymm11,0x60(%rdi)
vmovupd %ymm12,0x80(%rdi)
vmovupd %ymm13,0xa0(%rdi)
vmovupd %ymm14,0xc0(%rdi)
vmovupd %ymm15,0xe0(%rdi)
ret
.size	cplx_fft16_avx_fma, .-cplx_fft16_avx_fma
.section .note.GNU-stack,"",@progbits
