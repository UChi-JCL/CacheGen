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
        const auto m = static_cast<cdf_t>((left + right) / 2);
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
__device__ unsigned int hexCharToInt(char c) {
    if (c >= '0' && c <= '9') return (unsigned int)(c - '0');
    if (c >= 'a' && c <= 'f') return (unsigned int)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (unsigned int)(c - 'A' + 10);
    return 0; // Ideally, handle invalid character error
}

__device__ unsigned int parseHexToUint(const char* hexStr) {
    unsigned int result = 0;
    while (*hexStr) {
        result = result * 16 + hexCharToInt(*hexStr);
        hexStr++;
    }
    return result;
}

__device__ void deviceStrcpyFromIndex(char* dst, const char* src, int startIdx) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = src[startIdx + i];
    }
    // dst[8] = '\0'; // Null-terminate the destination string
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

// __host__ __device__  void f(
__global__ void decode_with_cuda_kernel(
    // int* out_arr, char* in, const int size_in, const cdf_t* cdf, const int N_sym, const int Lp ) {
    int *out_arr, char *in, const int size_in, cdf_t *cdf, const int N_sym, const int Lp, int *start_index_arr_device)
{
    int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    // char* tmpStart = (char*)malloc(8 * sizeof(char));
    // deviceStrcpyFromIndex(tmpStart, start_indices, start_index * 8);
    // uint from_index = parseHexToUint(tmpStart);
    // printf("start_index: %d, from_index: %d\n", start_index, from_index);
    int from_index = start_index_arr_device[start_index];
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

std::string formatIntegerTo4BitChars(unsigned int value) {
    std::string result;
    for (int i = 0; i < 8; ++i) { // Process each 4-bit nibble
        // Extract the rightmost nibble and convert it to a hex digit
        char hexDigit = "0123456789ABCDEF"[value & 0xF];
        // Prepend the digit to the result string
        result = hexDigit + result;
        // Shift the integer to get the next nibble
        value >>= 4;
    }
    return result;
}
// torch::Tensor decode(torch::Tensor out_tensor, const at::Tensor &cdf,
//                      std::vector<std::string> &stringList, const int all_tokens, 
//                      const int blockNum, const int threadNum)
// {
//     /* Decode a list of strings using the given CDF with CUDA kernels.
//      * Args:
//      *     out_tensor: A tensor of shape (all_tokens, N_sym) to store the decoded symbols, NEEDS to be on CUDA. 
//      *     cdf: A tensor of shape (1, N_sym, Lp) containing the CDF.
//      *     stringList: A list of bitstreams to decode.
//      *     all_tokens: The total number of tokens in the list of strings.
//      *     blockNum: The number of blocks to use for the kernel.
//      *     threadNum: The number of threads to use for the kernel.
//      * Returns:
//      *     A tensor of shape (all_tokens, N_sym) containing the decoded symbols. 
//     */

//     // const std::string& in) {
//     // int tot_length = concatenatedString.length();
//     std::cout << "all_tokens: " << all_tokens << std::endl;
//     int *start_index = new int[all_tokens];
//     start_index[0] = 0;
//     std::string concatenatedString;
//     int tot_length = 0;
//     int cnt = 0;

    
//     cudaEvent_t start3;
//     cudaEventCreate(&start3);
//     cudaEventRecord(start3, 0);
//     std::string concatStart;
//     float elapsedTime1;
//     for (const auto &str : stringList)
//     {
//         int start = concatenatedString.length();
//         concatenatedString += str;
//         int end = concatenatedString.length(); // End position is exclusive
//         start_index[cnt] = start;
//         // std::cout << "start: " << start_index[cnt]  << " end: " << end_index[cnt] << std::endl;
//         tot_length += str.length();
//         std::string formattedString = formatIntegerTo4BitChars(start);
//          // append format string to concatStart
//         concatStart += formattedString;
//         cnt += 1;
//     }
    

//     cudaEvent_t stop3;
//     cudaEventCreate(&stop3);
//     cudaEventRecord(stop3, 0);
//     cudaEventSynchronize(stop3);
//     cudaEventElapsedTime(&elapsedTime1, start3, stop3);
//     std::cout << "time taken for moving data: " << elapsedTime1 << " ms" << std::endl;
//     std::cout << "legnth of concatStart: " << concatenatedString.length() << std::endl;


//     char *d_str;
//     cudaMalloc((void **)&d_str, tot_length * sizeof(char));
//     cudaMemcpy(d_str, concatenatedString.c_str(), tot_length * sizeof(char), cudaMemcpyHostToDevice);


//     char *d_str_start;
//     cudaMalloc((void **)&d_str_start, concatStart.length() * sizeof(char));
//     cudaMemcpy(d_str_start, concatStart.c_str(),  concatStart.length()  * sizeof(char), cudaMemcpyHostToDevice);


    
//     cudaEvent_t start2, stop2;
//     cudaEventCreate(&start2);
//     cudaEventRecord(start2, 0);

//     const auto cdf_ptr = get_cdf_ptr_cuda(cdf);
//     // std::cout << "N_sym: " << cdf_ptr.N_sym << std::endl;

//     float elapsedTime;
//     // std::cout << "Start Length: " << tot_length << std::endl

    
//     cdf_t *cdf_data;
//     size_t size_cdf = cdf_ptr.N_sym * cdf_ptr.Lp * sizeof(cdf_t); // Calculate the size of the array.
//     cudaMalloc(&cdf_data, size_cdf);
//     cudaMemcpy(cdf_data, cdf_ptr.data, size_cdf, cudaMemcpyHostToDevice);
//     // std::cout << "nsymbols " << cdf_ptr.N_sym << std::endl;

    
//     // torch::Tensor out_tensor = torch::zeros(
//     //     {all_tokens, cdf_ptr.N_sym}, torch::kInt32).to(torch::kCUDA);
//     int *out_arr = out_tensor.data_ptr<int>();


//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventRecord(start, 0);


//     decode_with_cuda_kernel<<<blockNum, threadNum>>>(out_arr, d_str, tot_length, cdf_data, cdf_ptr.N_sym, cdf_ptr.Lp, d_str_start);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Kernel launch error: %s\n", cudaGetErrorString(err));
//     }
//     cudaEventCreate(&stop);
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);

//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     std::cout << "Time taken by decode_cuda kernels: " << elapsedTime << " ms" << std::endl;

//     cudaFree(d_str);
//     cudaFree(cdf_data);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return out_tensor;

// }

torch::Tensor decode_fast(torch::Tensor out_tensor, const at::Tensor &cdf,
                     std::string concated_string, std::vector<int> start_indices, const int all_tokens, 
                     const int blockNum, const int threadNum)
{
    /* Decode a list of strings using the given CDF with CUDA kernels.
     * Args:
     *     out_tensor: A tensor of shape (all_tokens, N_sym) to store the decoded symbols, NEEDS to be on CUDA. 
     *     cdf: A tensor of shape (1, N_sym, Lp) containing the CDF.
     *     stringList: A list of bitstreams to decode.
     *     all_tokens: The total number of tokens in the list of strings.
     *     blockNum: The number of blocks to use for the kernel.
     *     threadNum: The number of threads to use for the kernel.
     * Returns:
     *     A tensor of shape (all_tokens, N_sym) containing the decoded symbols. 
    */

    // const std::string& in) {
    // int tot_length = concatenatedString.length();
    int tot_length = 0;
    int cnt = 0;

    
    
  
    tot_length = concated_string.length();
    char *d_str;
    cudaMalloc((void **)&d_str, tot_length * sizeof(char));
    cudaMemcpy(d_str, concated_string.c_str() , tot_length * sizeof(char), cudaMemcpyHostToDevice);

    
    // cudaEvent_t start3;
    // cudaEventCreate(&start3);
    // cudaEventRecord(start3, 0);
    std::string concatStart;
    float elapsedTime1;
    int *start_index_arr_device;
    cudaMalloc((void **)&start_index_arr_device, start_indices.size() * sizeof(int));
    cudaMemcpy(start_index_arr_device, start_indices.data(), start_indices.size() * sizeof(int), cudaMemcpyHostToDevice);


    char *d_str_start;
    cudaMalloc((void **)&d_str_start, concatStart.length() * sizeof(char));
    cudaMemcpy(d_str_start, concatStart.c_str(),  concatStart.length()  * sizeof(char), cudaMemcpyHostToDevice);

    // cudaEvent_t stop3;
    // cudaEventCreate(&stop3);
    // cudaEventRecord(stop3, 0);
    // cudaEventSynchronize(stop3);
    // cudaEventElapsedTime(&elapsedTime1, start3, stop3);
    // std::cout << "time taken for moving data: " << elapsedTime1 << " ms" << std::endl;


    
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventRecord(start2, 0);

    const auto cdf_ptr = get_cdf_ptr_cuda(cdf);
    // std::cout << "N_sym: " << cdf_ptr.N_sym << std::endl;

    float elapsedTime;
    // std::cout << "Start Length: " << tot_length << std::endl

    
    cdf_t *cdf_data;
    size_t size_cdf = cdf_ptr.N_sym * cdf_ptr.Lp * sizeof(cdf_t); // Calculate the size of the array.
    cudaMalloc(&cdf_data, size_cdf);
    cudaMemcpy(cdf_data, cdf_ptr.data, size_cdf, cudaMemcpyHostToDevice);
    
    int *out_arr = out_tensor.data_ptr<int>();
    

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventRecord(start, 0);


    decode_with_cuda_kernel<<<blockNum, threadNum>>>(out_arr, d_str, tot_length, cdf_data, cdf_ptr.N_sym, cdf_ptr.Lp, start_index_arr_device);

    cudaEventCreate(&stop2);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime1, start2, stop2);
    std::cout << "time taken for compute data: " << elapsedTime << " ms" << std::endl;


    cudaFree(d_str);
    cudaFree(cdf_data);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    return out_tensor;

}



namespace py = pybind11;

PYBIND11_MODULE(torchac_cuda, m) {
    // m.def("decode", &decode, "decode function");
    m.def("decode_fast", &decode_fast, "Fast decode function");
}
