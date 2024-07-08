
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
from fastchat.model import load_model
p = argparse.ArgumentParser()
p.add_argument("--encoded_dir", type = str, default = "encoded", help="The directory to encoded kv. ")
p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
# p.add_argument("--kv_path", type = int, default = 0)
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
    # Check if encoded_dir is exists
    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    key_value = torch.load(f"{args.save_dir}/raw_kv_0.pt")
    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
    meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
    cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
    deserializer = CacheGenDeserializer(lmcache_config, meta_data)
    bytes = cachegen_serializer.to_bytes(key_value)
    pickle.dump(bytes, open(f"{args.encoded_dir}/doc_{args.doc_id}.pkl", "wb"))
    for _ in range(10):
        st = time.monotonic()
        decoded_kv = deserializer.from_bytes(bytes)
        torch.cuda.synchronize()
        print("Time to decode: ", time.monotonic() - st)
    