
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from src.cachegen_engine import CacheGenController


p = argparse.ArgumentParser()
p.add_argument("--path", type = str, default = "7k_prompts/1.txt", help="Path to the context file")
p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
# p.add_argument("--kv_path", type = int, default = 0)
p.add_argument("--generate_kv", action="store_true", default = None)
p.add_argument("--save_dir", type=str, default = None)
args = p.parse_args()
if __name__ == "__main__":
    # 
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Model and tokenizer loaded")
    os.environ['TMP_DIR'] = args.save_dir
    # Generate KV cache here 
    if args.generate_kv:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, load_in_8bit=True)
        model.eval()
        st = time.monotonic()
        with torch.no_grad():
            with open(args.path, "r") as f:
                text = f.read()
            input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
            # input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
            generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate = True)
        print(f"TTFT: {time.monotonic() - st}")
        pickle.dump(generated['past_key_values'], open(f"{args.save_dir}/test_kv.pkl", "wb"))
    else:
        with open(args.path, "r") as f:
            text = f.read()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        os.environ['TOKENS'] = str(input_ids.shape[1])
        cachegen_controller = CacheGenController(model=args.model_id)
        orig_kv_cache = pickle.load(open("data/test_kv.pkl", "rb"))
        
        
        cachegen_controller.set(input_ids[0], orig_kv_cache)
        # cachegen_controller.get(input_ids[0])
        
        for _ in range(1):
            st = time.monotonic()
            merged_kv = cachegen_controller.engine.merge_kv(input_ids=input_ids)
            print(f"Total time: {time.monotonic() - st}")
            
        output = cachegen_controller.engine.model.generate(input_ids=input_ids, \
            past_key_values=merged_kv, max_new_tokens=20)
        print(tokenizer.decode(output[0][-20:]))
        