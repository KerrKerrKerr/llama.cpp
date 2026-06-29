#pragma once

// DPP-fused warp reductions and hardware math intrinsics for gfx906
// (Vega 20 / MI50 / Radeon VII). Falls back to standard shuffle + add on
// other architectures.

#include "common.cuh"

#if defined(__gfx906__) && defined(GGML_USE_HIP)

static __device__ __forceinline__ float gfx906_add_xor1_f32(float x) {
    float r;
    asm volatile(
        "s_nop 4\n"
        "v_add_f32_dpp %0, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
        : "=v"(r) : "v"(x) : "memory"
    );
    return r;
}

static __device__ __forceinline__ float gfx906_add_xor2_f32(float x) {
    float r;
    asm volatile(
        "s_nop 1\n"
        "v_add_f32_dpp %0, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
        : "=v"(r) : "v"(x) : "memory"
    );
    return r;
}

// fused v_add_f32_dpp row_shl/row_shr does not produce a correct xor4 across all lanes
// (verified by PPL regression 7.2 -> 8484), so this step falls back to shuffle + add.
static __device__ __forceinline__ float gfx906_add_xor4_f32(float x) {
    int v_src = __float_as_int(x);
    int v_shuffled;
    asm volatile(
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=&v"(v_shuffled) : "v"(v_src) : "memory"
    );
    return x + __int_as_float(v_shuffled);
}

static __device__ __forceinline__ float gfx906_add_xor8_f32(float x) {
    float r;
    asm volatile(
        "s_nop 1\n"
        "v_add_f32_dpp %0, %1, %1 row_ror:8 row_mask:0xf bank_mask:0xf\n"
        : "=v"(r) : "v"(x) : "memory"
    );
    return r;
}

// DPP row operations cannot cross 16-lane boundaries, so xor16 uses ds_swizzle.
static __device__ __forceinline__ float gfx906_add_xor16_f32(float x) {
    int v = __float_as_int(x);
    int t;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(t) : "v"(v) : "memory"
    );
    return x + __int_as_float(t);
}

template<int width = 64>
static __device__ __forceinline__ float gfx906_warp_reduce_sum_f32(float x) {
    static_assert(width <= 64, "width must be <= wave size");
    if constexpr (width >= 2)  x = gfx906_add_xor1_f32(x);
    if constexpr (width >= 4)  x = gfx906_add_xor2_f32(x);
    if constexpr (width >= 8)  x += __shfl_xor(x, 4, width);
    if constexpr (width >= 16) x = gfx906_add_xor8_f32(x);
    if constexpr (width >= 32) x += __shfl_xor(x, 16, width);
    if constexpr (width == 64) x += __shfl_xor(x, 32, 64);
    return x;
}

static __device__ __forceinline__ float gfx906_exp_f32(float x) {
    constexpr float LOG2_E = 1.4426950408889634f;
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x * LOG2_E));
    return r;
}

#endif // defined(__gfx906__) && defined(GGML_USE_HIP)