/**
 * CGBN-based 1024-bit multiplication
 */

#pragma once

#include "common.cuh"

namespace vanity {
namespace cuda {

/**
 * CGBN Multiplication: 1024-bit × 1024-bit = 2048-bit
 * Requires 32 threads to cooperate (TPI=32)
 * Uses shared memory passed from caller
 */
static __device__ void bigint_mul_1024_cgbn(
    const uint32_t* __restrict__ p,
    const uint32_t* __restrict__ q,
    uint32_t* __restrict__ n,
    cgbn_mem_t<1024>* p_mem,
    cgbn_mem_t<1024>* q_mem,
    cgbn_mem_t<1024>* result_low_mem,
    cgbn_mem_t<1024>* result_high_mem,
    uint32_t instance_idx
) {
    // Get instance index within block (0-7 for 256 threads with TPI=32)
    uint32_t local_idx = threadIdx.x / TPI;

    cgbn_error_report_t *report = nullptr;
    mul_context_t ctx(cgbn_no_checks, report, instance_idx);
    mul_env_t env(ctx);

    typename mul_env_t::cgbn_t p_bn, q_bn;
    typename mul_env_t::cgbn_wide_t result_wide;

    // All 32 threads load data into shared memory
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        p_mem[local_idx]._limbs[i] = p[i];
        q_mem[local_idx]._limbs[i] = q[i];
    }

    // Load into CGBN (cooperative operation)
    cgbn_load(env, p_bn, &p_mem[local_idx]);
    cgbn_load(env, q_bn, &q_mem[local_idx]);

    // Multiply (cooperative operation - all 32 threads work together)
    cgbn_mul_wide(env, result_wide, p_bn, q_bn);

    // Store result (cooperative operation)
    cgbn_store(env, &result_low_mem[local_idx], result_wide._low);
    cgbn_store(env, &result_high_mem[local_idx], result_wide._high);

    // Only thread 0 of each group copies back to output
    if (threadIdx.x % TPI == 0) {
        for (uint32_t i = 0; i < PRIME_WORDS; i++) {
            n[i] = result_low_mem[local_idx]._limbs[i];
            n[PRIME_WORDS + i] = result_high_mem[local_idx]._limbs[i];
        }
    }
}

} // namespace cuda
} // namespace vanity
