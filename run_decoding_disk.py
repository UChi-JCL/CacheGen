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
from src.decode_interface import *
import argparse
p = argparse.ArgumentParser()
p.add_argument("--model_config", type = str, default = "config/mistral_7b.json", help="path to model config. ")
p.add_argument("--path_to_encoded_kv", type = str, default = "encoded/", help="The directory to encoded kv. ")
p.add_argument("--num_chunks", type = int, default = 4, help="number of chunks. ")
p.add_argument("--chunk_size", type = int, default = 2000, help="number of tokens in a chunk. ")
p.add_argument("--quantization_config", type = str, default = "config/quantization_7b.json", help="path to quantization config. ")
p.add_argument("--model_id", type = str, default = "mistral-community/Mistral-7B-v0.2")
p.add_argument("--path_to_context", type = str, default = "9k_prompts/0.txt", help="Path to the context file")
args = p.parse_args()

load_cache_stream = torch.cuda.Stream()



def _renorm_cast_cdf_(cdf, precision):
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf


def disk_load_worker(path, chunk_id, cdf_buffer, start_indices_buffer, bits_buffer, len_buffer):
    with torch.cuda.stream(load_cache_stream):
        encoded_file = pickle.load(open(f"{path}/{chunk_id}.pkl", "rb"))
        x = encoded_file['bitstreams']
        bits_buffer[:len(x)].copy_(x)
        start_indices = encoded_file['start_indices']
        start_indices = start_indices.pin_memory()
        start_indices_buffer.copy_(start_indices)
        len_buffer.copy_(torch.tensor([len(x)]).int())
        cdf_buffer.copy_(encoded_file["cdf"])
    
if __name__ == "__main__":
    path_to_encoded_kv = args.path_to_encoded_kv
    st = time.monotonic()
    pickle.load(open(f"{path_to_encoded_kv}/{0}.pkl", "rb"))
    print("Loading time: ", time.monotonic() - st)
    print("start")
    chunk_id = 0
    model_config = json.load(open(args.model_config, "r"))
    quantization_config = json.load(open(args.quantization_config, "r"))
    model, tokenizer = load_model(
                args.model_id,
                device="cuda",
                num_gpus=1,
                max_gpu_memory=f"40GiB",
                load_8bit=True,
                cpu_offloading=False,
                debug=False,
            )
    
    output = torch.zeros( (args.chunk_size, 2 * model_config['layers'] * model_config['channels'] )).to(torch.int).cuda()
    
    st = time.monotonic()
    encoded_file = pickle.load(open(f"{path_to_encoded_kv}/{chunk_id}.pkl", "rb"))
    max_tensors_key = encoded_file["max_tensors_key"]
    max_tensors_value = encoded_file["max_tensors_value"]
    cdf = encoded_file["cdf"]
    bits = encoded_file["bitstreams"]
    start_indices = encoded_file["start_indices"]
    start_indices = start_indices.pin_memory()
    start_indices = start_indices.cuda().int()
    cdf_buffer = torch.empty(cdf.shape).to(torch.int16)
    start_indices_buffer = torch.empty(start_indices.shape).int()
    bits_buffer = torch.empty(( 2 * model_config['layers'] * model_config['channels'] * args.chunk_size)).byte()
    len_buffer = torch.empty(1).long()
    inference_cdf = cdf
    inference_start_indices = start_indices
    inference_bits = torch.empty((2 * model_config['layers'] * model_config['channels'] * args.chunk_size)).byte()
    inference_bits[:len(bits)].copy_(bits)
    inference_len = torch.tensor([len(bits)]).int()
    final_kv = None
    for c in range(args.num_chunks):
        if c < args.num_chunks - 1:
            if c % 2 == 0:
                t1 = threading.Thread(target=disk_load_worker, 
                                      args=(args.path_to_encoded_kv, 
                                            c + 1,
                                            cdf_buffer, 
                                            start_indices_buffer, 
                                            bits_buffer, 
                                            len_buffer))
            else:
                t1 = threading.Thread(target=disk_load_worker, 
                                      args=(args.path_to_encoded_kv, 
                                            c + 1, 
                                            inference_cdf, 
                                            inference_start_indices, 
                                            inference_bits, 
                                            inference_len))
            t1.start()
        if c % 2 == 0:
            decoded  = decode_function(inference_cdf, 
                            inference_bits[:inference_len[0]], 
                            inference_start_indices, 
                            max_tensors_key, 
                            max_tensors_value, 
                            quantization_config, 
                            args.chunk_size, 
                            output,
                            c)
        else:
            decoded = decode_function(cdf_buffer, 
                            bits_buffer[:len_buffer[0]], 
                            start_indices_buffer, 
                            max_tensors_key, 
                            max_tensors_value, 
                            quantization_config, 
                            args.chunk_size, 
                            output,
                            c)
        
        final_kv = merge_kv(final_kv, decoded)
        if c < args.num_chunks - 1:
            t1.join()
            
    torch.cuda.synchronize() #sync + join
    temp_time = time.monotonic() - st
    print("end-to-end time: ", temp_time)
    

    with open(args.path_to_context, "r") as f:
        text = f.read()
        tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
        output = model.generate(tokens, max_new_tokens = 10, past_key_values=final_kv)
        print("output: ", tokenizer.decode(output[0, -10:]))