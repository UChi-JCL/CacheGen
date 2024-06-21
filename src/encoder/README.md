# CUDA-accelerated encoder

## Installation
Please run
```
cd src/encoder
pip install .
```

## Usage
Import the library through
```
import mytorchac
```

In most cases, you will only need the `encode_float_cdf()` function
```
output = mytorchac.encode_float_cdf(input_cdf, 
                                    input_symbols, 
                                    use_cuda = True, 
                                    max_out_size = 10000,
                                    blockNum = 1,
                                    threadNum = num_threads)
```
where the inputs and outputs are 
- `input_cdf` is the CDF that we would like to use during arithmetic encoding. Recommended data type is `torch.float32`, and every entry should be between 0.0 and 1.0 (inclusive). 
- `input_symbols` is the integer tensor that we would like to compress with arithmetic encoding. Recommended data type is `torch.int16` or `torch.int32`, and every entry should be larger than or equal to 0 (if not, please shift all entries by the same number so they are all non-negative). 
- `use_cuda`: If set to `True`, use CUDA-accelerated version of arithmetic encoding. If set to `False`, use the CPU version, i.e., `torchac.encode_float_cdf()`.
- `max_out_size`: This is the amount of GPU memory (in bytes) that we allocate for each CUDA thread for storing encoded outputs. We can safely set this to be a number slightly larger than the integer tensor size that each CUDA thread will be encoding.
- `blockNum`: The number of blocks. A block in CUDA is a collection of threads that execute the same kernel function.
- `threadNum`: The number of threads associated with each block.
- `output`: This will be a list of encoded outputs (corresponding to each integer tensor input, in order). The length of the list will be equal to the number of total threads.

Suggestions:
- Shaping: The first two dimensions of `input_cdf` and `input_symbols` should always align. In one of my test cases, the shape of `input_cdf` is `torch.Size([1000, 1024, 33])`; The shape of `input_symbols` is `torch.Size([1000, 1024])`, indicating that I have 1000 integer tensors (each of size 1024) to encode.
- Number of threads to use: It is generally recommended that you use one thread for each integer tensor input that you would like to encode; otherwise the results would be different from the CPU-encoded ones. In my case, as `input_symbols` has shape `torch.Size([1000, 1024])`, I use 1000 threads. For now, you can safely set `blockNum` to be 1 and `threadNum` to be the total number of threads that you would like to use. Later on, you can adjust these two parameters accordingly when you want more sophisticated parallelism, e.g., using multiple blocks could be more reasonable for layer-wise parallelism. 


## Example
Please try the `CacheGen/run_encoding_cuda.py` example as a sanity check.