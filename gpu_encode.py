
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
os.environ['HF_TOKEN'] = "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"
from src.cachegen_engine import CacheGenController
import json
from torch.profiler import profile, record_function, ProfilerActivity
from fastchat.model import load_model
p = argparse.ArgumentParser()
p.add_argument("--encoded_dir", type = str, default = "encoded", help="The directory to encoded kv. ")
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

def tensor_to_tuple(kv):
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0), 
                       kv[i][1].unsqueeze(0)))
    return tuple(new_kv)

if __name__ == "__main__":
    
    key_value = torch.load(f"{args.save_dir}/test_kv_0.pt")
    lmcache_config = LMCacheEngineConfig.from_defaults()
    meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
    cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
    deserializer = CacheGenDeserializer(lmcache_config, meta_data)
    ntokens = key_value.shape[-2]
    combined_decoded_kv = []
    for i in range(ntokens//256):
        st = time.monotonic()
        bytes = cachegen_serializer.to_bytes(key_value[:, :, :, i*256:(i+1)*256])
        torch.cuda.synchronize()
        # print("Time to encode: ", time.monotonic() - st)
        st = time.monotonic()
        decoded_kv = deserializer.from_bytes(bytes)
        combined_decoded_kv += [decoded_kv]
        torch.cuda.synchronize()
        torch.save(torch.frombuffer(bytes, dtype=torch.uint8),f"{args.encoded_dir}/doc_{0}_{i}.pt")
        # pickle.dump(bytes, open(f"{args.encoded_dir}/doc_{0}_{i}.pkl", "wb"))
        # print("Time to decode: ", time.monotonic() - st)
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