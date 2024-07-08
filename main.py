
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
import json
from fastchat.model import load_model
from src.utils import *


p = argparse.ArgumentParser()

p.add_argument("--path", type = str, default = "7k_prompts/1.txt", help="Path to the context file")
p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
# p.add_argument("--kv_path", type = int, default = 0)
p.add_argument("--generate_kv", action="store_true", default = None)
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--vanilla", action="store_true", default = None)
p.add_argument("--doc_id", type=int, default = 0)
p.add_argument("--results_dir", type=str, default = "results")
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--quantization_config", type=str, default="config/quantization.json")
p.add_argument("--path_to_context", type=str)
args = p.parse_args()
if __name__ == "__main__":
    model, tokenizer = load_model(
            args.model_id,
            device="cuda",
            num_gpus=args.num_gpus,
            max_gpu_memory=f"{args.max_gpu_memory}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )
    # model = AutoModelForCausalLM.from_pretrained(args.model_id, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Model and tokenizer loaded")
    os.environ['TMP_DIR'] = args.save_dir
    # Generate KV cache here 
    
    with open(args.path_to_context, "r") as f:
        text = f.read()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    print("Length of input: ", input_ids.shape)
    st = time.monotonic()
    
    generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
    torch.cuda.synchronize()
    print( f"TTFT: {time.monotonic() - st}" )
    kv = generated['past_key_values']
    kv = list(kv)
    key_value = []
    for i in range(len(kv)):
        kv[i] = list(kv[i])
        kv[i][0] = kv[i][0][:, :, :-1][0]
        kv[i][1] = kv[i][1][:, :, :-1][0]
        kv[i] = tuple(kv[i])
    kv = tuple(kv)
    kv_tensor = to_blob(kv)
    torch.save(kv_tensor, f"{args.save_dir}/raw_kv_{args.doc_id}.pt")
    