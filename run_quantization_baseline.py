
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
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
p.add_argument("--bins", type=int)
p.add_argument("--results_dir", type=str, default = None)
p.add_argument("--results_str", type=str, default = "gt")
args = p.parse_args()
if __name__ == "__main__":
    # Check if save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    # model = AutoModelForCausalLM.from_pretrained(args.model_id, load_in_8bit=True)
    print("Model and tokenizer loaded")
    # Generate KV cache here 
    for doc_id in range(args.start, args.end):
        kv = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        kv = default_quantization(kv, args.bins)
        with open(f"{args.path_to_context}/{doc_id}.txt", "r") as f:
            text = f.read()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        generated = model.generate(input_ids, past_key_values=kv, max_new_tokens = 20)
        with open(f"{args.results_dir}/{args.results_str}_{doc_id}.txt", "w") as f:
            f.write(tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))
        print(tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))