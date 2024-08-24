
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
import json
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from src.utils import *


p = argparse.ArgumentParser()
p.add_argument("--slo", type=float, default=2)
p.add_argument("--chunk_size", type=int, default=1000)
p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--dataset_name", type=str)
p.add_argument("--encode", action="store_true")
p.add_argument("--total_traces", type=int, default=5)
p.add_argument("--calculate_metric", type=int, default=0)
args = p.parse_args()
if __name__ == "__main__":
    # Check if save_dir exists
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    data = load_testcases(DATASET_TO_PATH[args.dataset_name])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if args.encode:
        for doc_id in range(args.start, args.end):
            print("Saving KV cache for doc: ", doc_id)
            text = data[doc_id]['prompt']
            input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
            # print("Length of input: ", input_ids.shape)
            st = time.monotonic()
            
            generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
            torch.cuda.synchronize()
            kv = generated['past_key_values']
            kv = list(kv)
            key_value = []
            for i in range(len(kv)):
                kv[i] = list(kv[i])
                kv[i][0] = kv[i][0][:, :, :-1][0]
                kv[i][1] = kv[i][1][:, :, :-1][0]
                kv[i] = tuple(kv[i])
            kv = tuple(kv)
            pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{doc_id}.pkl", "wb"))
            kv_tensor = to_blob(kv)
            num_chunks = (input_ids.shape[-1] // args.chunk_size)
            chunk_id = 0
            for chunk_start in range(0, input_ids.shape[-1], args.chunk_size):
                
                for quant_level in range(1, 4):
                    os.environ["QUANT_LEVEL"] = str(quant_level)
                    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=args.chunk_size)
                    meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
                    cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
                    bytestreams = cachegen_serializer.to_bytes(kv_tensor[:, :, :, chunk_start:chunk_start + args.chunk_size])
                    pickle.dump(bytestreams, open(f"{args.save_dir}/{doc_id}_{chunk_id}_{quant_level}.pkl", "wb"))
                chunk_id += 1
            
                
    else:

        chunk_delay = 0.2
        traces = load_testcases("test_data/bw.jsonl")
        average_violation_rate = []
        average_acc = []
        for bw_id in range(args.total_traces):
            
            all_bws = traces[bw_id]['bw']
            violation_rate = 0
            
            for doc_id in range(args.start, args.end):
                layer_to_device_id = {}
                kv = pickle.load(open(f"{args.save_dir}/raw_kv_{doc_id}.pkl", "rb"))
                for i in range(len(kv)):
                    layer_to_device_id[i] = kv[i][0].device.index
                text = data[doc_id]['prompt']
                input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
                ttft, configs = config_selection(all_bws, 
                                            chunk_delay, 
                                            args, 
                                            input_ids.shape[-1],
                                            doc_id,
                                            )
                concated_kv = merge(configs, args, doc_id, input_ids.shape[-1], kv, layer_to_device_id)
                adapt_output = model.generate(input_ids, past_key_values=concated_kv, max_new_tokens=20)
                prediction = tokenizer.decode(adapt_output[0][input_ids.shape[1]:], skip_special_tokens=True)
                # print(prediction)
                if ttft > args.slo:
                    violation_rate += 1
                if args.calculate_metric == 1:
                    if args.dataset_name == "longchat":
                        metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                        average_acc += [metric]
                    elif args.dataset_name == "nqa" or args.dataset_name == "tqa":
                        metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                        average_acc += [metric]
            # print("Violation rate: ", violation_rate/(args.end - args.start))
            average_violation_rate += [violation_rate/(args.end - args.start)]
            print("Average violation rate: ", sum(average_violation_rate)/len(average_violation_rate))
            print("Average accuracy: ", sum(average_acc)/len(average_acc))
        print("Average violation rate: ", sum(average_violation_rate)/len(average_violation_rate))