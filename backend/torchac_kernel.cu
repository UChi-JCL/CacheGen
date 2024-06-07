/*
 * COPYRIGHT 2019 ETH Zurich
 */
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
const int precision = 16;
const int N = 1;
using cdf_t = uint16_t;
const int PRECISION = 16;
const int RENORMALIZATION_FACTOR = 2 << (PRECISION - 1);
const int STRIDE = 1;


__host__ __device__ cdf_t binsearch_cuda(cdf_t *cdf, cdf_t target, cdf_t max_sym,
                                         const int offset) /* i * Lp */
{
    cdf_t left = 0;
    cdf_t right = max_sym + 1; // len(cdf) == max_sym + 2

    while (left + 1 < right)
    { // ?
        // left and right will be < 0x10000 in practice, so left+right fits in uint16_t...
        const auto m = static_cast<const cdf_t>((left + right) / 2);
        const auto v = cdf[offset + m];
        // printf("Index: %d\n", offset + m);
        if (v < target)
        {
            left = m;
        }
        else if (v > target)
        {
            right = m;
        }
        else
        {
            return m;
        }
    }

    return left;
}


struct cdf_ptr
{
    const cdf_t *data; // expected to be a N_sym x Lp matrix, stored in row major.
    const int N_sym;   // Number of symbols stored by `data`.
    const int Lp;      // == L+1, where L is the number of possible values a symbol can take.
    cdf_ptr(const cdf_t *data,
            const int N_sym,
            const int Lp) : data(data), N_sym(N_sym), Lp(Lp){};
};


/** Get an instance of the `cdf_ptr` struct. */
const struct cdf_ptr get_cdf_ptr(const torch::Tensor& cdf)
{
    TORCH_CHECK(!cdf.is_cuda(), "cdf must be on CPU!")
    const auto s = cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected (N, Lp)")

    const int N_sym = s[0];
    const int Lp = s[1];
    const auto cdf_acc = cdf.accessor<int16_t, 2>();
    const cdf_t* cdf_ptr = (uint16_t*)cdf_acc.data();

    const struct cdf_ptr res(cdf_ptr, N_sym, Lp);
    return res;
}


const struct cdf_ptr get_cdf_ptr_cuda(const at::Tensor &cdf)
{
    // AT_CHECK(!cdf.is_cuda(), "cdf must be on CPU!")
    const auto s = cdf.sizes();
    // AT_CHECK(s.size() == 4 && s[0] == 1, "Invalid size for cdf! Expected 1HWLp")

    const int N_sym = s[1] * s[2];
    const int Lp = s[3];
    const auto cdf_reshaped = at::reshape(cdf, {N_sym, -1});
    const auto cdf_acc = cdf_reshaped.accessor<int16_t, 2>();
    const cdf_t *cdf_ptr = (uint16_t *)cdf_acc.data();

    const struct cdf_ptr res(cdf_ptr, N_sym, Lp);
    return res;
}


__device__ void append_cache_to_string(char* out, uint8_t& cache, int max_out_size) {
    // find the end of the string
    int length = 0;
    while (length < max_out_size - 1 && out[length] != '\0') {
        length++;
    }

    // append the new character if there's space
    if (length < max_out_size - 1) {
        out[length] = static_cast<char>(cache);
        out[length + 1] = '\0';
    }
}


__device__ void append_to_end(char* out, uint8_t cache, int* current_out_length) {
    // find the end of the string
    out[*current_out_length] = static_cast<char>(cache);

    // append the null character
    out[*current_out_length + 1] = '\0';
}


__device__ void append_to_end_uint(char* out, uint8_t cache, 
                                   uint16_t device_out_offset, uint32_t current_out_length) {
    // find the end of the string
    out[device_out_offset + current_out_length] = static_cast<char>(cache);

    // append the null character
    out[device_out_offset + current_out_length + 1] = '\0';
}


// // cuda version (single-threaded)
// __global__ void encode_with_cuda_naive(int16_t* device_sym, 
//                                        char* device_out, 
//                                        const int max_out_size,
//                                        const cdf_t *cdf, 
//                                        const int N_sym, 
//                                        const int Lp, 
//                                        int* current_out_length)
// {
//     printf("enter encode_with_cuda_naive()\n");

//     printf("N_sym = %d\n", N_sym);

//     int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    
//     uint8_t cache = 0;
//     uint8_t count = 0;
//     int bit = 0;

//     uint32_t low = 0;
//     uint32_t high = 0xFFFFFFFFU;
//     uint64_t pending_bits = 0;

//     const int precision = 16;

//     const int max_symbol = Lp - 2;

//     for (int i = 0; i < N_sym; ++i) {
//         const int16_t sym_i = device_sym[i];

//         const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

//         const int offset = i * Lp;
//         // Left boundary is at offset + sym_i
//         const uint32_t c_low = cdf[offset + sym_i];
//         // Right boundary is at offset + sym_i + 1, except for the `max_symbol`
//         // For which we hardcode the maxvalue. So if e.g.
//         // L == 4, it means that Lp == 5, and the allowed symbols are
//         // {0, 1, 2, 3}. The max symbol is thus Lp - 2 == 3. It's probability
//         // is then given by c_max - cdf[-2].
//         const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

//         high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
//         low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

//         while (true) {
//             if (high < 0x80000000U) {
//                 // out_cache.append_bit_and_pending(0, pending_bits);
//                 bit = 0;
//                 cache <<= 1;
//                 cache |= bit;
//                 count += 1;
//                 if (count == 8) {
//                     // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                     // append_cache_to_string(device_out, cache, max_out_size);
//                     append_to_end(device_out, cache, current_out_length);
//                     *current_out_length += 1;
//                     // printf("Value after increment: %d\n", *current_out_length);
//                     count = 0;
//                 }
//                 while (pending_bits > 0) {
//                     bit = 1;
//                     cache <<= 1;
//                     cache |= bit;
//                     count += 1;
//                     if (count == 8) {
//                         // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                         // append_cache_to_string(device_out, cache, max_out_size);
//                         append_to_end(device_out, cache, current_out_length);
//                         *current_out_length += 1;
//                         // printf("Value after increment: %d\n", *current_out_length);
//                         count = 0;
//                     }
//                     pending_bits -= 1;
//                 }

//                 low <<= 1;
//                 high <<= 1;
//                 high |= 1;
//             } else if (low >= 0x80000000U) {
//                 // out_cache.append_bit_and_pending(1, pending_bits);
//                 bit = 1;
//                 cache <<= 1;
//                 cache |= bit;
//                 count += 1;
//                 if (count == 8) {
//                     // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                     // append_cache_to_string(device_out, cache, max_out_size);
//                     append_to_end(device_out, cache, current_out_length);
//                     *current_out_length += 1;
//                     // printf("Value after increment: %d\n", *current_out_length);
//                     count = 0;
//                 }
//                 while (pending_bits > 0) {
//                     bit = 0;
//                     cache <<= 1;
//                     cache |= bit;
//                     count += 1;
//                     if (count == 8) {
//                         // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                         // append_cache_to_string(device_out, cache, max_out_size);
//                         append_to_end(device_out, cache, current_out_length);
//                         *current_out_length += 1;
//                         // printf("Value after increment: %d\n", *current_out_length);
//                         count = 0;
//                     }
//                     pending_bits -= 1;
//                 }

//                 low <<= 1;
//                 high <<= 1;
//                 high |= 1;
//             } else if (low >= 0x40000000U && high < 0xC0000000U) {
//                 pending_bits++;
//                 low <<= 1;
//                 low &= 0x7FFFFFFF;
//                 high <<= 1;
//                 high |= 0x80000001;
//             } else {
//                 break;
//             }
//         }
//     }

//     pending_bits += 1;

//     if (pending_bits) {
//         if (low < 0x40000000U) {
//             // out_cache.append_bit_and_pending(0, pending_bits);
//             bit = 0;
//             cache <<= 1;
//             cache |= bit;
//             count += 1;
//             if (count == 8) {
//                 // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                 // append_cache_to_string(device_out, cache, max_out_size);
//                 append_to_end(device_out, cache, current_out_length);
//                 *current_out_length += 1;
//                 // printf("Value after increment: %d\n", *current_out_length);
//                 count = 0;
//             }
//             while (pending_bits > 0) {
//                 bit = 1;
//                 cache <<= 1;
//                 cache |= bit;
//                 count += 1;
//                 if (count == 8) {
//                     // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                     // append_cache_to_string(device_out, cache, max_out_size);
//                     append_to_end(device_out, cache, current_out_length);
//                     *current_out_length += 1;
//                     // printf("Value after increment: %d\n", *current_out_length);
//                     count = 0;
//                 }
//                 pending_bits -= 1;
//             }
//         } else {
//             // out_cache.append_bit_and_pending(1, pending_bits);
//             bit = 1;
//             cache <<= 1;
//             cache |= bit;
//             count += 1;
//             if (count == 8) {
//                 // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                 // append_cache_to_string(device_out, cache, max_out_size);
//                 append_to_end(device_out, cache, current_out_length);
//                 *current_out_length += 1;
//                 // printf("Value after increment: %d\n", *current_out_length);
//                 count = 0;
//             }
//             while (pending_bits > 0) {
//                 bit = 0;
//                 cache <<= 1;
//                 cache |= bit;
//                 count += 1;
//                 if (count == 8) {
//                     // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                     // append_cache_to_string(device_out, cache, max_out_size);
//                     append_to_end(device_out, cache, current_out_length);
//                     *current_out_length += 1;
//                     // printf("Value after increment: %d\n", *current_out_length);
//                     count = 0;
//                 }
//                 pending_bits -= 1;
//             }
//         }
//     }

//     // flush
//     if (count > 0) {
//         for (int i = count; i < 8; ++i) {
//             // append(0);
//             bit = 0;
//             cache <<= 1;
//             cache |= bit;
//             count += 1;
//             if (count == 8) {
//                 // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
//                 // append_cache_to_string(device_out, cache, max_out_size);
//                 append_to_end(device_out, cache, current_out_length);
//                 *current_out_length += 1;
//                 // printf("Value after increment: %d\n", *current_out_length);
//                 count = 0;
//             }
//         }
//         assert(count==0);
//     }

// }


// cuda version (multi-threaded)
__global__ void encode_with_cuda(int16_t* device_sym, 
                                 char* device_out, 
                                 const uint16_t max_out_size,
                                 const cdf_t *cdf, 
                                 const int N_sym, 
                                 const int Lp, 
                                 uint32_t* device_out_lengths,
                                 const uint16_t total_num_of_threads) {
    
    // printf("enter encode_with_cuda()\n");

    // printf("N_sym = %d\n", N_sym);
    
    uint8_t cache = 0;
    uint8_t count = 0;
    int bit = 0;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;

    const int precision = 16;

    const int max_symbol = Lp - 2;

    // multi-threading related variables
    uint16_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t per_thread_sym_coverage = ceil(N_sym / total_num_of_threads);
    const uint16_t sym_offset = thread_index * per_thread_sym_coverage;
    const uint16_t device_out_offset = thread_index * max_out_size;

    // for (int i = 0; i < N_sym; ++i) {
    for (int i = sym_offset; i < sym_offset + per_thread_sym_coverage; ++i) {

        const int16_t sym_i = device_sym[i];

        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        const int offset = i * Lp;
        // Left boundary is at offset + sym_i
        const uint32_t c_low = cdf[offset + sym_i];
        // Right boundary is at offset + sym_i + 1, except for the `max_symbol`
        // For which we hardcode the maxvalue. So if e.g.
        // L == 4, it means that Lp == 5, and the allowed symbols are
        // {0, 1, 2, 3}. The max symbol is thus Lp - 2 == 3. It's probability
        // is then given by c_max - cdf[-2].
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (high < 0x80000000U) {
                // out_cache.append_bit_and_pending(0, pending_bits);
                bit = 0;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                while (pending_bits > 0) {
                    bit = 1;
                    cache <<= 1;
                    cache |= bit;
                    count += 1;
                    if (count == 8) {
                        // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                        // append_cache_to_string(device_out, cache, max_out_size);
                        append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                        atomicAdd(&device_out_lengths[thread_index], 1);
                        // printf("Value after increment: %d\n", *current_out_length);
                        count = 0;
                    }
                    pending_bits -= 1;
                }

                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x80000000U) {
                // out_cache.append_bit_and_pending(1, pending_bits);
                bit = 1;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                while (pending_bits > 0) {
                    bit = 0;
                    cache <<= 1;
                    cache |= bit;
                    count += 1;
                    if (count == 8) {
                        // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                        // append_cache_to_string(device_out, cache, max_out_size);
                        append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                        atomicAdd(&device_out_lengths[thread_index], 1);
                        // printf("Value after increment: %d\n", *current_out_length);
                        count = 0;
                    }
                    pending_bits -= 1;
                }

                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            } else {
                break;
            }
        }
    }

    pending_bits += 1;

    if (pending_bits) {
        if (low < 0x40000000U) {
            // out_cache.append_bit_and_pending(0, pending_bits);
            bit = 0;
            cache <<= 1;
            cache |= bit;
            count += 1;
            if (count == 8) {
                // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                // append_cache_to_string(device_out, cache, max_out_size);
                append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                atomicAdd(&device_out_lengths[thread_index], 1);
                // printf("Value after increment: %d\n", *current_out_length);
                count = 0;
            }
            while (pending_bits > 0) {
                bit = 1;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                pending_bits -= 1;
            }
        } else {
            // out_cache.append_bit_and_pending(1, pending_bits);
            bit = 1;
            cache <<= 1;
            cache |= bit;
            count += 1;
            if (count == 8) {
                // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                // append_cache_to_string(device_out, cache, max_out_size);
                append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                atomicAdd(&device_out_lengths[thread_index], 1);
                // printf("Value after increment: %d\n", *current_out_length);
                count = 0;
            }
            while (pending_bits > 0) {
                bit = 0;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                pending_bits -= 1;
            }
        }
    }

    // flush
    if (count > 0) {
        for (int i = count; i < 8; ++i) {
            // append(0);
            bit = 0;
            cache <<= 1;
            cache |= bit;
            count += 1;
            if (count == 8) {
                // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                // append_cache_to_string(device_out, cache, max_out_size);
                append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                atomicAdd(&device_out_lengths[thread_index], 1);
                // printf("Value after increment: %d\n", *current_out_length);
                count = 0;
            }
        }
        assert(count==0);
    }

}


// __host__ __device__  void f(
__global__ void decode_with_cuda_kernel(
    // int* out_arr, char* in, const int size_in, const cdf_t* cdf, const int N_sym, const int Lp ) {
    int *out_arr, char *in, const int size_in, cdf_t *cdf, const int N_sym, const int Lp, const int *start_indices, const int *end_indices)
{
    int start_index = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("start_index: %d \n", start_index);
    int from_index = start_indices[start_index]; // The start index of the string in the concatenated string 'in'
    int to_index = end_indices[start_index];     // The end index of the string in the concatenated string 'in'
    // printf("start_index: %d\n", start_index);
    // char* dynamicArray = (char*)malloc((to_index - from_index ) * sizeof(char));
    // // printf("from_index: %d, to_index: %d\n", from_index, to_index);
    // for (int i = from_index; i < to_index  ; i++) {
    //     dynamicArray[i - from_index] = in[i];
    // }
    const int max_symbol = Lp - 2;

    int out = 0;
    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint32_t value = 0;
    const uint32_t c_count = 0x10000U;
    const int precision = 16; // TODO: unify with torchac_kernel.cu

    uint8_t cache = 0;
    uint8_t cached_bits = 0; // num
    size_t in_ptr = 0;

    // printf("strt!!!!!!!!\n");
    for (int i = 0; i < 32; i++)
    {
        if (cached_bits == 0)
        {
            if (in_ptr == size_in)
            {
                value <<= 1;
                continue;
            }
            /// Read 1 byte
            const int index = in_ptr;
            cache = (uint8_t)in[index + from_index];

            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits--;
        // get(in, &cache, &cached_bits, &in_ptr, &value);
    }
    //  printf("strt3!!!!!!!!\n");

    for (int i = 0; i < N_sym; ++i)
    {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        // always < 0x10000 ???
        const uint16_t count = ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) * c_count - 1) / span;
        // printf("strt2!!!!!!!!\n");
        const int offset = i * Lp;
        auto sym_i = binsearch_cuda(cdf, count, (cdf_t)max_symbol, offset);
        out_arr[(blockIdx.x * blockDim.x + threadIdx.x ) * N_sym + i] = (int16_t)sym_i;
        // out_arr[i] = (int16_t)sym_i;

        // if (i == N_sym-1) {
        //     break;
        // }
        // if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        //     printf("GPU_sym_i: %d\n", sym_i);
        // }
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];
        // printf("c_low: %d, c_high: %d\n", c_low, c_high);
        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);
        while (true)
        {
            if (low >= 0x80000000U || high < 0x80000000U)
            {
                low <<= 1;
                high <<= 1;
                high |= 1;
                if (cached_bits == 0)
                {
                    if (in_ptr == size_in)
                    {
                        value <<= 1;
                        continue;
                    }
                    /// Read 1 byte
                    cache = (uint8_t)in[in_ptr + from_index];
                    in_ptr++;
                    cached_bits = 8;
                }
                value <<= 1;
                value |= (cache >> (cached_bits - 1)) & 1;
                cached_bits--;
            }
            else if (low >= 0x40000000U && high < 0xC0000000U)
            {
                /**
                 * 0100 0000 ... <= value <  1100 0000 ...
                 * <=>
                 * 0100 0000 ... <= value <= 1011 1111 ...
                 * <=>
                 * value starts with 01 or 10.
                 * 01 - 01 == 00  |  10 - 01 == 01
                 * i.e., with shifts
                 * 01A -> 0A  or  10A -> 1A, i.e., discard 2SB as it's all the same while we are in
                 *    near convergence
                 */
                low <<= 1;
                low &= 0x7FFFFFFFU; // make MSB 0
                high <<= 1;
                high |= 0x80000001U; // add 1 at the end, retain MSB = 1
                value -= 0x40000000U;
                if (cached_bits == 0)
                {
                    if (in_ptr == size_in)
                    {
                        value <<= 1;
                        continue;
                    }
                    /// Read 1 byte
                    cache = (uint8_t)in[in_ptr + from_index];
                    in_ptr++;
                    cached_bits = 8;
                }
                value <<= 1;
                value |= (cache >> (cached_bits - 1)) & 1;
                cached_bits--;

                // get(&in_cache, value);
            }
            else
            {
                break;
            }
        }
    }

    // return out_arr;
}


// __global__ void decode_cuda(const cdf_ptr& cdf_ptr,
//          char* in, const int size_in) {
//     printf("HELLLO \n");
//    f(cdf_ptr, in, size_in);
// }


namespace
{
    __device__ __forceinline__ float sigmoidf(float a)
    {
        return 1.0 / (1.0 + expf(-a));
    }

    __device__ __forceinline__ cdf_t renorm(float cdf, const int Lp, const int l)
    {
        cdf *= (RENORMALIZATION_FACTOR - (Lp - 1));
        cdf_t cdf_int = static_cast<cdf_t>(lrintf(cdf) + l);
        return cdf_int;
    }

    __global__ void calculate_cdf_kernel(
        const int N, const int Lp, const int K,
        const float *__restrict__ targets, // Lp length vector
        const float *__restrict__ means,
        const float *__restrict__ log_scales,
        const float *__restrict__ logit_probs_softmax,
        cdf_t *__restrict__ cdf_mem /* out */)
    {
        /**
         * Expects to be launched on a N*Lp grid? TODO
         *
         * means, log_scales, logit_probs_softmax:
         *      each is a 1KHW matrix reshaped to KN, where N = H*W
         * cdf_mem:
         *      an array of length N * Lp, representing a NxLp matrix, where
         *      cdf[n][l] = cdf_mem[n*Lp + l]
         *
         * Code:
         *      for n, l in range(N) x range(Lp)
         *          target = l
         *          cdf_n_l = 0;
         *          for k in range(K)
         *              log_scale = log_scales[k][n]
         *              mean = means[k][n]
         *              logit_prob = logit_probs_softmax[k][n]
         *              inv_stdv = exp(log_scale)
         *              centered_target = target - mean
         *              cdf_n_l += logit_prob * sigmoid(centered_target * inv_stdv)
         *           cdf[n][l] = cdf_mem
         */
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i_1d = index; i_1d < N * Lp; i_1d += stride)
        {
            const int n = i_1d / Lp;
            const int l = i_1d % Lp;

            const float target = targets[l];
            float cdf_n_l_float = 0; // initialize

            for (int k = 0; k < K; ++k)
            {
                const float log_scale = log_scales[k * N + n];
                const float mean = means[k * N + n];
                const float logit_prob = logit_probs_softmax[k * N + n];
                const float inv_stdv = expf(-log_scale);
                const float centered_target = target - mean;
                cdf_n_l_float += logit_prob * sigmoidf(centered_target * inv_stdv);
            }

            const int cdf_n_l_idx = i_1d;
            cdf_mem[cdf_n_l_idx] = renorm(cdf_n_l_float, Lp, l);
        }
    }
}


cdf_t *malloc_cdf(const int N, const int Lp)
{
    cdf_t *cdf_mem;
    cudaMallocManaged(&cdf_mem, N * Lp * sizeof(cdf_t));
    return cdf_mem;
}


void free_cdf(cdf_t *cdf_mem)
{
    cudaFree(cdf_mem);
}


template <typename T>
std::string to_string(const T &object)
{
    std::ostringstream ss;
    ss << object;
    return ss.str();
}


// py::bytes encode_cuda_naive(const at::Tensor &cdf, 
//                             const at::Tensor &input_sym, 
//                             int blockNum, 
//                             int threadNum)
// {
//     // const std::string& in) {
//     // int tot_length = concatenatedString.length();
    
//     // cudaEvent_t start1, stop1;
//     // cudaEventCreate(&start1);
//     // cudaEventRecord(start1, 0);
//     // float elapsedTime1;
//     // float elapsedTime;

//     std::cout << "enter encode_cuda_naive()" << std::endl;

//     // allocate device memory for cdf
//     // const auto cdf_ptr = get_cdf_ptr_cuda(cdf);
//     const auto cdf_ptr = get_cdf_ptr(cdf);
//     cdf_t *cdf_data;
//     size_t size_cdf = cdf_ptr.N_sym * cdf_ptr.Lp * sizeof(cdf_t); // Calculate the size of the array.
//     cudaMalloc(&cdf_data, size_cdf);
//     cudaMemcpy(cdf_data, cdf_ptr.data, size_cdf, cudaMemcpyHostToDevice);
    
//     // allocate device memory for device_sym (AC input)
//     int16_t* device_sym;
//     size_t device_sym_size = input_sym.numel() * sizeof(int16_t);
//     cudaMalloc(&device_sym, device_sym_size);
//     cudaMemcpy(device_sym, input_sym.data_ptr<int16_t>(), device_sym_size, cudaMemcpyHostToDevice);

//     // allocate device memory for device_out (AC output)
//     char* device_out;
//     const int max_out_size = 10000;
//     size_t device_out_size = max_out_size * sizeof(char);
//     cudaMalloc(&device_out, device_out_size);
//     cudaMemset(device_out, 0, device_out_size);

//     // allocate device memory for current_out_length
//     int* device_current_out_length;
//     cudaMalloc(&device_current_out_length, sizeof(int));
//     cudaMemset(device_current_out_length, 0, sizeof(int));

//     // torch::Tensor out_tensor = torch::zeros(
//     //     {all_tokens, cdf_ptr.N_sym}, torch::kInt32).to(torch::kCUDA);
//     // int *out_arr = out_tensor.data_ptr<int>();

//     // Copy the start and end indices to the GPU
//     // int *device_start_index;
//     // int *device_end_index;
//     // cudaMalloc(&device_start_index, all_tokens * sizeof(int));
//     // cudaMalloc(&device_end_index, all_tokens * sizeof(int));
//     // cudaMemcpy(device_start_index, start_index, all_tokens * sizeof(int), cudaMemcpyHostToDevice);
//     // cudaMemcpy(device_end_index, end_index, all_tokens * sizeof(int), cudaMemcpyHostToDevice);
    
//     // cudaEventCreate(&stop1);
//     // cudaEventRecord(stop1, 0);
//     // cudaEventSynchronize(stop1);
//     // cudaEventElapsedTime(&elapsedTime1, start1, stop1);
//     // std::cout << "time taken to move data: " << elapsedTime1 << " ms" << std::endl;

//     // cudaEvent_t start, stop;
//     // cudaEventCreate(&start);
//     // cudaEventRecord(start, 0);

//     blockNum = 1;
//     threadNum = 1;

//     encode_with_cuda_naive<<<blockNum, threadNum>>>(device_sym, device_out, max_out_size, cdf_data, cdf_ptr.N_sym, cdf_ptr.Lp, device_current_out_length);

//     // for debugging
//     int local_current_out_length = 0;
//     cudaMemcpy(&local_current_out_length, device_current_out_length, sizeof(int), cudaMemcpyDeviceToHost);
//     std::cout << "local_current_out_length = " << local_current_out_length << std::endl;

//     // move encoded data back to main memory
//     std::vector<char> host_out(max_out_size);
//     cudaMemcpy(host_out.data(), device_out, device_out_size, cudaMemcpyDeviceToHost);
    
//     // cudaEventCreate(&stop);
//     // cudaEventRecord(stop, 0);
//     // cudaEventSynchronize(stop);

//     // cudaEventElapsedTime(&elapsedTime, start, stop);
//     // cudaFree(d_str);
//     // cudaFree(cdf_data);
//     // cudaFree(device_start_index);
//     // cudaFree(device_end_index);
//     // std::cout << "Time taken by decode_cuda kernels: " << elapsedTime << " ms" << std::endl;

//     // cudaEventDestroy(start);
//     // cudaEventDestroy(stop);

//     // return out_tensor;
    
//     cudaFree(cdf_data);
//     cudaFree(device_sym);
//     cudaFree(device_out);

//     return py::bytes(host_out.data(), local_current_out_length);
// }


py::bytes encode_cuda(const at::Tensor &cdf, 
                      const at::Tensor &input_sym, 
                      const uint16_t max_out_size,
                      const int blockNum, 
                      const int threadNum)
{
    // const std::string& in) {
    // int tot_length = concatenatedString.length();
    
    // cudaEvent_t start1, stop1;
    // cudaEventCreate(&start1);
    // cudaEventRecord(start1, 0);
    // float elapsedTime1;
    // float elapsedTime;

    // const uint16_t max_out_size = 10000;

    const uint16_t total_num_of_threads = blockNum * threadNum;

    // allocate device memory for cdf
    // const auto cdf_ptr = get_cdf_ptr_cuda(cdf);
    // std::cout << "allocate device memory for cdf" << std::endl;
    const auto cdf_ptr = get_cdf_ptr(cdf);
    cdf_t *cdf_data;
    const size_t size_cdf = cdf_ptr.N_sym * cdf_ptr.Lp * sizeof(cdf_t); // Calculate the size of the array.
    cudaMalloc(&cdf_data, size_cdf);
    cudaMemcpy(cdf_data, cdf_ptr.data, size_cdf, cudaMemcpyHostToDevice);
    
    // allocate device memory for device_sym (AC input)
    // std::cout << "allocate device memory for device_sym (AC input)" << std::endl;
    int16_t* device_sym;
    const size_t device_sym_size = input_sym.numel() * sizeof(int16_t);
    cudaMalloc(&device_sym, device_sym_size);
    cudaMemcpy(device_sym, input_sym.data_ptr<int16_t>(), device_sym_size, cudaMemcpyHostToDevice);

    // allocate device memory for device_out (AC output)
    // NOTE: need separate memory for each thread
    // std::cout << "allocate device memory for device_out (AC output)" << std::endl;
    char* device_out;
    const size_t device_out_size = max_out_size * total_num_of_threads * sizeof(char);
    cudaMalloc(&device_out, device_out_size);
    cudaMemset(device_out, 0, device_out_size);

    // allocate device memory for device_out_lengths
    // NOTE: need separate memory for each thread
    // std::cout << "allocate device memory for device_out_lengths" << std::endl;
    uint32_t* device_out_lengths;
    const size_t device_out_lengths_size = total_num_of_threads * sizeof(uint32_t);
    cudaMalloc(&device_out_lengths, device_out_lengths_size);
    cudaMemset(device_out_lengths, 0, device_out_lengths_size);

    // torch::Tensor out_tensor = torch::zeros(
    //     {all_tokens, cdf_ptr.N_sym}, torch::kInt32).to(torch::kCUDA);
    // int *out_arr = out_tensor.data_ptr<int>();

    // Copy the start and end indices to the GPU
    // int *device_start_index;
    // int *device_end_index;
    // cudaMalloc(&device_start_index, all_tokens * sizeof(int));
    // cudaMalloc(&device_end_index, all_tokens * sizeof(int));
    // cudaMemcpy(device_start_index, start_index, all_tokens * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(device_end_index, end_index, all_tokens * sizeof(int), cudaMemcpyHostToDevice);
    
    // cudaEventCreate(&stop1);
    // cudaEventRecord(stop1, 0);
    // cudaEventSynchronize(stop1);
    // cudaEventElapsedTime(&elapsedTime1, start1, stop1);
    // std::cout << "time taken to move data: " << elapsedTime1 << " ms" << std::endl;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventRecord(start, 0);

    // std::cout << "before entering encode_with_cuda()" << std::endl;
    encode_with_cuda<<<blockNum, threadNum>>>(device_sym, 
                                              device_out, 
                                              max_out_size, 
                                              cdf_data, 
                                              cdf_ptr.N_sym, 
                                              cdf_ptr.Lp, 
                                              device_out_lengths,
                                              total_num_of_threads);
    // std::cout << "after returning from encode_with_cuda()" << std::endl;

    // for debugging
    uint32_t* local_out_lengths = new uint32_t[device_out_lengths_size];
    cudaMemcpy(local_out_lengths, device_out_lengths, device_out_lengths_size, cudaMemcpyDeviceToHost);

    // std::cout << "printing local_out_lengths" << std::endl;
    // for (uint16_t thread_index = 0; thread_index < total_num_of_threads; thread_index++) {
    //     std::cout << local_out_lengths[thread_index] << " ";
    // }
    // std::cout << std::endl;

    // move encoded data back to main memory
    std::vector<char> host_out(device_out_size);
    cudaMemcpy(host_out.data(), device_out, device_out_size, cudaMemcpyDeviceToHost);
    
    // concatenate valid results from different threads
    std::vector<char> valid_out;
    uint32_t valid_total_length = 0;
    for (uint16_t thread_index = 0; thread_index < total_num_of_threads; thread_index++) {
        uint32_t valid_length = local_out_lengths[thread_index];
        valid_total_length += valid_length;

        // append to valid_out
        valid_out.insert(valid_out.end(), host_out.begin() + thread_index * max_out_size, host_out.begin() + thread_index * max_out_size + valid_length);
    }

    // cudaEventCreate(&stop);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);

    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // cudaFree(d_str);
    // cudaFree(cdf_data);
    // cudaFree(device_start_index);
    // cudaFree(device_end_index);
    // std::cout << "Time taken by decode_cuda kernels: " << elapsedTime << " ms" << std::endl;

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // return out_tensor;
    
    cudaFree(cdf_data);
    cudaFree(device_sym);
    cudaFree(device_out);
    
    // std::cout << "before returning py::bytes()" << std::endl;

    // return py::bytes(host_out.data(), local_out_lengths[0]);
    return py::bytes(valid_out.data(), valid_total_length);
}


torch::Tensor decode_cuda(const at::Tensor &cdf,
                          std::vector<std::string> &stringList, 
                          const int all_tokens, 
                          const int blockNum, 
                          const int threadNum)
{
    // const std::string& in) {
    // int tot_length = concatenatedString.length();
    std::cout << "all_tokens: " << all_tokens << std::endl;
    int *start_index = new int[all_tokens];
    int *end_index = new int[all_tokens];
    start_index[0] = 0;
    std::string concatenatedString;
    int tot_length = 0;
    int cnt = 0;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventRecord(start1, 0);
    float elapsedTime1;
    for (const auto &str : stringList)
    {
        int start = concatenatedString.length();
        concatenatedString += str;
        int end = concatenatedString.length(); // End position is exclusive
        end_index[cnt] = end;
        start_index[cnt] = start;
        // std::cout << "start: " << start_index[cnt]  << " end: " << end_index[cnt] << std::endl;
        tot_length += str.length();
        cnt += 1;
    }

    char *d_str;
    cudaMalloc((void **)&d_str, tot_length * sizeof(char));
    cudaMemcpy(d_str, concatenatedString.c_str(), tot_length * sizeof(char), cudaMemcpyHostToDevice);

    const auto cdf_ptr = get_cdf_ptr_cuda(cdf);
    std::cout << "N_sym: " << cdf_ptr.N_sym << std::endl;

    float elapsedTime;
    // std::cout << "Start Length: " << tot_length << std::endl

    cdf_t *cdf_data;
    size_t size_cdf = cdf_ptr.N_sym * cdf_ptr.Lp * sizeof(cdf_t); // Calculate the size of the array.
    cudaMalloc(&cdf_data, size_cdf);
    cudaMemcpy(cdf_data, cdf_ptr.data, size_cdf, cudaMemcpyHostToDevice);
    // std::cout << "nsymbols " << cdf_ptr.N_sym << std::endl;


    torch::Tensor out_tensor = torch::zeros(
        {all_tokens, cdf_ptr.N_sym}, torch::kInt32).to(torch::kCUDA);
    int *out_arr = out_tensor.data_ptr<int>();

    // Copy the start and end indices to the GPU
    int *device_start_index;
    int *device_end_index;
    
    cudaMalloc(&device_start_index, all_tokens * sizeof(int));
    cudaMalloc(&device_end_index, all_tokens * sizeof(int));
    cudaMemcpy(device_start_index, start_index, all_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_end_index, end_index, all_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventCreate(&stop1);
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsedTime1, start1, stop1);
    std::cout << "time taken to move data: " << elapsedTime1 << " ms" << std::endl;



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);


    decode_with_cuda_kernel<<<blockNum, threadNum>>>(out_arr, d_str, tot_length, cdf_data, cdf_ptr.N_sym, cdf_ptr.Lp, device_start_index, device_end_index);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaFree(d_str);
    cudaFree(cdf_data);
    cudaFree(device_start_index);
    cudaFree(device_end_index);
    std::cout << "Time taken by decode_cuda kernels: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // for (int i = 0; i < 10; i++) {
    //     std::cout << "Out: " << hostArray[i] << std::endl;
    // }

    return out_tensor;

}

namespace py = pybind11;

PYBIND11_MODULE(mytorchac_cuda, m) {
    // m.def("encode", &encode, "encode function");
    // m.def("encode_with_cpu", &encode_with_cpu, "encode with cpu function");
    // m.def("encode_cpu", &encode_cpu, "encode with cpu function");
    m.def("encode_cuda", &encode_cuda, "encode with cuda function");
    m.def("decode_cuda", &decode_cuda, "decode with cuda function");
    // m.def("decode", &decode, "decode function");
    // m.def("decode_cuda", &decode_cuda, "decode_cuda function");
    // m.def("concat_str", &concat_str, "concat_str function");
}
// PYBIND11_MODULE(torchac, m) {
//     m.def("decode_cuda", &decode_cuda, "decode_cuda function");
// }