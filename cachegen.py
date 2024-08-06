
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import os
import time
import pickle
import torch
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward
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
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", action="store_true")

args = p.parse_args()


if __name__ == "__main__":
    # Check if encoded_dir is exists
    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    # Check if results_dir is exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    # Read data from jsonl
    data =  load_testcases(DATASET_TO_PATH[args.dataset_name])
    
    # replace_llama_forward_with_reuse_forward()
    kv_tokens = []
    # Start encoding
    layer_to_device_id = {}
    kv = pickle.load(open(f"{args.save_dir}/raw_kv_{args.start}.pkl", "rb"))
    for i in range(len(kv)):
        layer_to_device_id[i] = kv[i][0].device.index
    for doc_id in range(args.start, args.end):
        key_value = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
        meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
        cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
        bytes = cachegen_serializer.to_bytes(key_value)
        pickle.dump(bytes, open(f"{args.encoded_dir}/{doc_id}.pkl", "wb"))
        kv_tokens += [key_value.shape[-2]]
        
    # Start inferencing 
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    average_acc = []
    for doc_id in range(args.start, args.end):
        os.environ['DOC_ID'] = str(doc_id)
        print("Running inference for doc_id: ", doc_id)
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_tokens[doc_id])
        meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
        deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        bytes = pickle.load(open(f"{args.encoded_dir}/{doc_id}.pkl", "rb"))
        st = time.monotonic()
        decoded_kv = deserializer.from_bytes(bytes)
        torch.cuda.synchronize()
        print( f"TTFT: {time.monotonic() - st}" )
        decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
        text = data[doc_id]['prompt']
        # print(torch.nn.MSELoss()(decoded_kv[30][0][0], raw[30][0]))
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        with open(f"{args.results_dir}/{args.results_str}_{doc_id}.txt", "w") as f:
            f.write(prediction)
        metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
        average_acc += [metric]
        print(prediction)
    print("Average accuracy is: ", np.mean(average_acc))