
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from src.cachegen_engine import CacheGenController
import json
from torch.profiler import profile, record_function, ProfilerActivity
from fastchat.model import load_model
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

args = p.parse_args()
if __name__ == "__main__":
    # 
    model, tokenizer = load_model(
            args.model_id,
            device="cuda",
            num_gpus=args.num_gpus,
            max_gpu_memory=f"{args.max_gpu_memory}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Model and tokenizer loaded")
    os.environ['TMP_DIR'] = args.save_dir
    # Generate KV cache here 
    if args.generate_kv:
        with open(args.path, "r") as f:
            text = f.read()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        st = time.monotonic()
        # with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     # with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     # input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
        #     generated = model.generate(input_ids, max_new_tokens = 1)
        generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
        torch.cuda.synchronize()
        print( f"TTFT: {time.monotonic() - st}" )
        kv = generated['past_key_values']
        kv = list(kv)
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0].cpu()
            kv[i][1] = kv[i][1].cpu()
            kv[i] = tuple(kv[i])
        kv = tuple(kv)
        pickle.dump(kv, open(f"{args.save_dir}/test_kv.pkl", "wb"))
    elif args.vanilla:
        # model = AutoModelForCausalLM.from_pretrained(args.model_id, load_in_8bit=True)
        # model.eval()
        
        with open(args.path, "r") as f:
            text = f.read()
        
        for _ in range(5):
            # st = time.monotonic()
            with torch.no_grad():
                input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
                generated = model.generate(input_ids, max_new_tokens = 2)
            # print( f"TTFT: {time.monotonic() - st}" )
        print("output: ", tokenizer.decode(generated[0][-10:]))
        # Dump the generated output to output directory
        with open(f"{args.results_dir}/output_{args.doc_id}.txt", "w") as f:
            f.write(tokenizer.decode(generated[0][-10:]))
    else:
        with open(args.path, "r") as f:
            text = f.read()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        os.environ['TOKENS'] = str(input_ids.shape[1])
        quantization_config = json.load(open(args.quantization_config, "r"))
        
        cachegen_controller = CacheGenController(model=model, config=quantization_config)
        orig_kv_cache = pickle.load(open(f"{args.save_dir}/test_kv.pkl", "rb"))
        
        cachegen_controller.set(input_ids[0], orig_kv_cache, quantization_config)
        cachegen_controller.set(input_ids[0], orig_kv_cache, quantization_config, offline=False)
        # cachegen_controller.get(input_ids[0])
        
        for _ in range(1):
            st = time.monotonic()
            merged_kv = cachegen_controller.engine.merge_kv(input_ids=input_ids)
            print(f"Total time: {time.monotonic() - st}")
        # input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()[:, :4001]
        # output = cachegen_controller.engine.model.generate(input_ids=input_ids, \
        #     past_key_values=merged_kv, max_new_tokens=20)
        # print(tokenizer.decode(output[0][-20:]))
        