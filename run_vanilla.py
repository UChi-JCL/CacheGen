
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
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward

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
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", type=int)
args = p.parse_args()
if __name__ == "__main__":
    # Check if save_dir exists
    
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    data =  load_testcases(DATASET_TO_PATH[args.dataset_name])
    layer_to_device_id = {}
    average_acc = []
    for doc_id in range(args.start, args.end):
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        generated = model.generate(input_ids,  max_new_tokens = 20)
        prediction = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"doc id: {doc_id}", tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))
        if args.calculate_metric == 1:
            if args.dataset_name == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                average_acc += [metric]
            elif args.dataset_name == "nqa" or args.dataset_name == "tqa":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                average_acc += [metric]
    if args.calculate_metric == 1:
        if args.dataset_name == "longchat":
            print("Average vanilla accuracy is: ", np.mean(average_acc))
        elif args.dataset_name == "nqa" or args.dataset_name == "tqa":
            print("Average vanilla F1 score is: ", np.mean(average_acc))