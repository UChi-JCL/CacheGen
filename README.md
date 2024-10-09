# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

**For the latest update and integration, please check out the [LMCache](https://github.com/LMCache/LMCache) project!**

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf) (SIGCOMM'24). 
The code structure is organized as follows:

- ```LMCache```: The modules for KV cache encoding / decoding with CacheGen's customized codec 
- ```test_data```: The example testing cases for CacheGen. 
- ```src```: Some helper functions used by CacheGen (e.g., transforming tensor to tuple, transforming tuple to tensor etc.)

## Installation

To install the required **python** packages to run CacheGen with conda
```
conda env create -f env.yaml
conda activate cachegen
pip install -e LMCache
cd LMCache/third_party/torchac_cuda 
python setup.py install
```

### Examples 

Please refer to the page [sigcomm_ae.md](sigcomm_ae.md) for running examples for CacheGen. 

### Contact 
Yuhan Liu (yuhanl@uchicago.edu)
