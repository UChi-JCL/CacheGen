# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf).

## Installation

1. To install the required **python** packages to run CacheGen with conda
```
conda env create -f env.yml 
```
2. Build the GPU-version Arithmetic Coding (AC) encoder
```
cd src/encoder
pip install .
```
3. Build the GPU-version Arithmetic Coding (AC) decoder 
```
cd src/decoder
python setup.py install
```

We will combine steps #2 and #3 above very soon.

## Example run

First generate KV caches with the following command
```
 python main.py --model_id <MODEL_ID> --save_dir <PATH TO ORIGINAL KV>  --path_to_context 9k_prompts/0.txt --doc_id 0
 ```
An example run is:

```
python main.py \
--model_id mistral-community/Mistral-7B-v0.2 \
--save_dir data \
--path_to_context 9k_prompts
```

Next, run the encoding of KV caches with the following command
```
python run_encoding.py \
--num_chunks <TOTAL NUMBER OF CHUNKS> \
--output_path <PATH TO ENCODED KVs> \
--path_to_kv <PATH TO ORIGINAL KV> \
--quantization_config <PATH TO QUANTIZATION CONFIG>
```

An example run can be:

```
python run_encoding.py \
--num_chunks 4 \
--output_path encoded \
--path_to_kv data/test_kv_0.pkl \
--quantization_config config/quantization_7b.json
```

Finally, run the actual inference (and loading encoded KV caches) 

```
python run_decoding_disk.py \
--model_config <PATH TO MODEL CONFIG> \
--path_to_encoded_kv <PATH TO ENCODED KVs> \
--num_chunks <TOTAL NUMBER OF CHUNKS> \
--quantization_config <PATH TO QUANTIZATION CONFIG> \
--model_id <MODEL ID> \
--path_to_context <PATH TO context>
```

An example run is:
```
python run_decoding_disk.py \
--model_config config/mistral_7b.json \
--path_to_encoded_kv encoded \
--num_chunks 4 \
--quantization_config config/quantization_7b.json \
--model_id mistral-community/Mistral-7B-v0.2 \
--path_to_context 9k_prompts/0.txt
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
