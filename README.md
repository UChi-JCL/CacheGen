# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf).

This branch is for standalone runs of CacheGen's encoder and decoder (for Jiayi or Kuntai's integration)

## Installation

1. To install the required **python** packages to run CacheGen with conda
```
conda env create -f env.yml 
```
2. Build the GPU-version Arithmetic Coding (AC) decoder 
```
cd src/decoder
python setup.py install
```

GPU-version AC encoder will come soon!


## Example run
To generate the KV cache given a text file, run
```
LAYERS=32 CHANNELS=4096 python main.py --generate_kv --path 9k_prompts/1.txt --save_dir <PATH TO YOUR HOME DIRECTORY> --model_id <MODEL YOU WANT TO RUN>
```

To run encoding
```
python test_encoder.py \
--path_to_encoded_kv <PATH TO ENCODED KV> \
--quantization_config <PATH TO QUANTIZATION CONFIG> \
--model_config <PATH TO MODEL CONFIG> \
--path_to_original_kv <PATH TO ORIGINAL KV > \
--chunk_size <CHUNK SIZE> 
```

An example:
```
python test_encoder.py \
--path_to_encoded_kv data/test_encoded.pkl \
--quantization_config config/quantization_7b.json \
--model_config config/model_config.json \
--path_to_original_kv data/test_kv.pkl \
--chunk_size 2000 
```

Note that to define a model config, you should include the model's hidden dimension (e.g., 4096 for Llama 7B, 1024 for Mistral 7B or Llama 70B), # of heads and the # of dimensions per head

An example for Llama 7B:

```
{
    "hidden_dim": 4096, 
    "num_heads": 32,
    "heads_dim": 128
}
```

To run decoding
```
python test_decoder.py \
--path_to_encoded_kv <PATH TO ENCODED KV, NEEDS TO BE THE SAME AS THE PATH DURING ENCODING> \
--quantization_config <PATH TO QUANTIZATION CONFIG> \
--model_config <PATH TO MODEL CONFIG> \
--input_text <PATH TO TEXT> \
--chunk_size <CHUNK SIZE> \
--model_id <MODEL YOU WANT TO RUN>
```

An example:
```
python test_decoder.py \
--path_to_encoded_kv data/test_encoded.pkl \
--quantization_config config/quantization_7b.json \
--model_config config/model_config.json \
--input_text 7k_prompts/2.txt \
--chunk_size 2000 \
--model_id lmsys/longchat-7b-16k 
```



## References

```
@misc{liu2024cachegen,
      title={CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming}, 
      author={Yuhan Liu and Hanchen Li and Yihua Cheng and Siddhant Ray and Yuyang Huang and Qizheng Zhang and Kuntai Du and Jiayi Yao and Shan Lu and Ganesh Ananthanarayanan and Michael Maire and Henry Hoffmann and Ari Holtzman and Junchen Jiang},
      year={2024},
      eprint={2310.07240},
      archivePrefix={arXiv},
      primaryClass={cs.NI}
}
```

## FAQs
