# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf).

## Installation

1. To install the required **python** packages to run CacheGen with conda
```
conda env create --name cachegenenv --file env.yml
```
2. Build the GPU-version Arithmetic Coding (AC) decoder 
```
cd src/decoder
python setup.py install
```

GPU-version AC encoder will come soon!


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
