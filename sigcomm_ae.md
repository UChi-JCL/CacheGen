### Encode/Decoding KV caches 
Before running the encoder of CacheGen, first runs the following code to generate the KV caches. The following code generates the KV caches for the LongChat dataset (in Fig. 8/9) on Mistral-7B model.
Note that this assumes you have one A40 GPU with 48GB GPU memeory. 
```
bash scripts/generate.sh
```

Then run the KV cache encoding and decoding code:
```
bash scripts/cachegen.sh
```

