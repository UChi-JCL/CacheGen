from llmlingua import PromptCompressor
import argparse
import os
import pickle
from fastchat.model import load_model
parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--start', type=int, required=True, 
                    help='an optional argument to clear something')
parser.add_argument('--end', type=int, required=True, 
                    help='an optional argument to clear something')
parser.add_argument('--input_dir', type=str, required=True, 
                    help='an optional argument to clear something')
parser.add_argument('--target_tokens', type=int, required=True, 
                    help='an optional argument to clear something')
parser.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
parser.add_argument("--output_dir", type=str, default = "results")
parser.add_argument("--active", action="store_true", default = None)
args = parser.parse_args()

if __name__ == "__main__":
    
    
    if args.active:
        llm_lingua = PromptCompressor()
        for i in range(args.start, args.end):
            prompt = f"{args.input_dir}/{i}.txt"
            with open(prompt, "r") as f:
                prompt = f.read()
            compressed_prompt = llm_lingua.compress_prompt(prompt, instruction="", \
                question="", target_token=args.target_tokens)
            prompt = compressed_prompt["compressed_prompt"]
            with open(f"{args.output_dir}/llmlingua_input_{i}.txt", "w") as f:
                f.write(prompt)
            # input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # output = model.generate(input_tokens, max_new_tokens = 10, return_dict_in_generate=True)
            # breakpoint()
    else:
        model, tokenizer = load_model(
            args.model_id,
            device="cuda",
            num_gpus=1,
            # max_gpu_memory=f"{args.max_gpu_memory}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )
        
        for i in range(args.start, args.end):
            prompt = f"{args.output_dir}/llmlingua_input_{i}.txt"
            with open(prompt, "r") as f:
                prompt = f.read()
            input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            print("Length of input tokens", len(input_tokens[0]))
            output = model.generate(input_tokens, max_new_tokens = 20, return_dict_in_generate=True)
            with open(f"{args.output_dir}/llmlingua_results_{i}.txt", "w") as f:
                f.write(tokenizer.decode(output["sequences"][0][len(input_tokens[0]):], skip_special_tokens=True))
            pickle.dump(output['past_key_values'], open(f"{args.output_dir}/llmlingua_kv_{i}.pkl", "wb"))