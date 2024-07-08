import torch
import time
import threading
import pickle
import torchac_cuda
import os
import json
import requests
import io

from fastchat.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from src.decode_interface import *
import argparse
p = argparse.ArgumentParser()
p.add_argument("--num_gpus", type = int, default = 1)
p.add_argument("--max_gpu_memory", type = int, default = 40, help="Default max GPU memory in GiB on A40")
p.add_argument("--model_config", type = str, default = "config/mistral_7b.json", help="path to model config. ")
p.add_argument("--path_to_encoded_kv", type = str, default = "encoded/", help="The directory to encoded kv. ")
p.add_argument("--num_chunks", type = int, default = 4, help="number of chunks. ")
p.add_argument("--chunk_size", type = int, default = 2000, help="number of tokens in a chunk. ")
p.add_argument("--quantization_config", type = str, default = "config/quantization_7b.json", help="path to quantization config. ")
p.add_argument("--model_id", type = str, default = "mistral-community/Mistral-7B-v0.2")
p.add_argument("--path_to_context", type = str, default = "9k_prompts_instruct/0.txt", help="Path to the context file")
args = p.parse_args()
def tensor_to_tuple(kv):
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0), 
                       kv[i][1].unsqueeze(0)))
    return tuple(new_kv)
def disk_load_worker(path, chunk_id, bytes_buffer):
    with torch.cuda.stream(load_cache_stream):
        bytes = torch.load(f"{path}/doc_0_{chunk_id}.pt")
        # bytes_bufferbytes
        # copuy bytes to buffer
        bytes_buffer[:len(bytes)].copy_(bytes)
load_cache_stream = torch.cuda.Stream()
if __name__ == "__main__":
    
    lmcache_config = LMCacheEngineConfig.from_defaults()
    meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
    cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
    deserializer = CacheGenDeserializer(lmcache_config, meta_data)
    even_bytes = torch.empty(( 64 * 1024 * 256)).byte()
    odd_bytes = torch.empty(( 64 * 1024 * 256)).byte()
    
    loaded_bytes = torch.load(f"{args.path_to_encoded_kv}/doc_0_0.pt")
    even_bytes[:len(loaded_bytes)].copy_(loaded_bytes)
    decode_even = even_bytes.numpy().tobytes()
    decoded_kv = deserializer.from_bytes(decode_even)
    st = time.monotonic()
    loaded_bytes = torch.load(f"{args.path_to_encoded_kv}/doc_0_0.pt")
    even_bytes[:len(loaded_bytes)].copy_(loaded_bytes)
    combined_decoded_kv = []
    for c in range(1, 30):
        if c < 2:
            if c % 2 == 0:
                t1 = threading.Thread(target=disk_load_worker, 
                                      args=(args.path_to_encoded_kv, 
                                            c, 
                                            even_bytes)) #(args.path_to_encoded_kv, c, even_bytes)
                
            else:
                t1 = threading.Thread(target=disk_load_worker, 
                                      args=(args.path_to_encoded_kv, 
                                            c, 
                                            odd_bytes))
            t1.start()
        if c % 2 == 0:
            decode_odd = odd_bytes.numpy().tobytes()
            decoded_kv = deserializer.from_bytes(decode_odd)
        else:
            decode_even = even_bytes.numpy().tobytes()
            decoded_kv = deserializer.from_bytes(decode_even)
        combined_decoded_kv.append(decoded_kv)
        print(decoded_kv.shape)
        t1.join()
    torch.cuda.synchronize()
    print("Time to load: ", time.monotonic() - st)
    combined_decoded_kv = torch.cat(combined_decoded_kv, dim=3)
    decoded_kv_tuple = tensor_to_tuple(combined_decoded_kv)
    
    model, tokenizer = load_model(
            args.model_id,
            device="cuda",
            num_gpus=args.num_gpus,
            max_gpu_memory=f"{args.max_gpu_memory}GiB",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
    with open(args.path_to_context, "r") as f:
        text = f.read()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids, past_key_values=decoded_kv_tuple, max_new_tokens=10)
    print(tokenizer.decode(output[0][-10:], skip_special_tokens=True))