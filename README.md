# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf). 
The code structure is organized as follows:

- ```LMCache```: The modules for KV cache encoding / decoding with CacheGen's customized codec 
- ```test_data```: The example testing cases for CacheGen. 


## Installation

To install the required **python** packages to run CacheGen with conda
```
conda env create -f env.yaml
pip install -e LMCache
cd LMCache/third_party/torchac_cuda 
python setup.py install
```

### Examples 

Please refer to the page [sigcomm_ae.md](sigcomm_ae.md) for running examples for CacheGen. 

### Contact 
Yuhan Liu (yuhanl@uchicago.edu), Yihua Cheng (yihua98@uchicago.edu) 