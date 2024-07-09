
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
import json
from src.utils import *

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--encoded_dir", type=str, default = None)
p.add_argument("--results_dir", type=str, default = None)
p.add_argument("--results_str", type=str, default = "results")
args = p.parse_args()



if __name__ == "__main__":
    # Check if encoded_dir is exists
    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    # Check if results_dir is exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    model, tokenizer = load_model(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    for doc_id in range(args.start, args.end):
        key_value = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
        meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
        cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
        deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        bytes = cachegen_serializer.to_bytes(key_value)
        pickle.dump(bytes, open(f"{args.encoded_dir}/{doc_id}.pkl", "wb"))
        decoded_kv = deserializer.from_bytes(bytes)
        decoded_kv = tensor_to_tuple(decoded_kv)
        with open(f"{args.path_to_context}/{doc_id}.txt", "r") as f:
            text = f.read()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
        print(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
        with open(f"{args.results_dir}/{args.results_str}_{doc_id}.txt", "w") as f:
            f.write(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
    