import torch 
import pickle
import os
import torchac_cuda
import time
import json
import argparse
from fastchat.model import load_model
p = argparse.ArgumentParser()
p.add_argument("--path_to_encoded_kv", type=str)
p.add_argument("--quantization_config", type=str)
p.add_argument("--model_config", type=str)
p.add_argument("--chunk_size", type=int)
p.add_argument("--model_id", type=str)
p.add_argument("--input_text", type=str)
args = p.parse_args()
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
def quant(xq, max1, dim=-1, quant_type="vector"):
    
    C = int(os.environ["BINS"]) // 2 - 1
    x = (xq / C * max1).to(torch.float16)
    return x

def transformer_kv_to_tuple( key, value, block, thread):
    """ Field:
    - key: a torch.Tensor of shape (l, N, 4096)
    """
    kv_list = []
    for i in range(len(key)):
        
        tmp1 = key[i].reshape((key[i].shape[0], block, thread)).permute((1, 0, 2)).unsqueeze(0)
        tmp2 = value[i].reshape((key[i].shape[0], block, thread)).permute((1, 0, 2)).unsqueeze(0)
        kv_list += [(tmp1, tmp2)]
    return tuple(kv_list)


def decode_function(path_to_encoded_kv, quantization_config, model_config, CHUNK_SIZE):
    """
    Given the path to the encoded key value cache, decode the KV cache
    Fields:
    - path_to_encoded_kv: the path to the encoded key value cache
    - quantization_config: the path to the quantization config
    - model_config: the path to the model config
    - CHUNK_SIZE: the chunk size to decode, NEEDS to be multiples of 20!!! 
    Outputs:
    - key: the decoded key tensor in the shape of (layers, num_heads, tokens, heads_dim)
    """
    config = json.load(open(quantization_config, "r"))
    model_config = json.load(open(model_config, "r"))
    encoded_file = pickle.load(open(path_to_encoded_kv, "rb"))
    cdf = encoded_file["cdf"]
    cdf = _renorm_cast_cdf_(cdf.float(), 16)
    output = torch.zeros( (CHUNK_SIZE, cdf.shape[0] * model_config['hidden_dim'] )).cuda().to(torch.int)
    bits = encoded_file["bitstreams"]
    concated_string = bits
    start_indices= encoded_file["start_indices"]
    max_tensors_k = encoded_file["max_tensors_key"]
    max_tensors_v = encoded_file["max_tensors_value"]
    for i in range(2): # 2 times to warm up the cache
        st = time.monotonic()
        out = torchac_cuda.decode_fast(output, cdf.unsqueeze(0), concated_string, \
            start_indices, CHUNK_SIZE, 20, CHUNK_SIZE//20)
        # out = torchac_cuda.decode(output, cdf.unsqueeze(0), bits,  6000, 60, 100)
        print( f"TTFT: {time.monotonic() - st}")
    out = output.reshape((CHUNK_SIZE, 2, max_tensors_k.shape[0], \
        model_config["hidden_dim"])).permute(1, 2, 0, 3)
    key = out[0].half()
    value = out[1].half()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            os.environ['BINS'] = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            os.environ['BINS'] = config["key_second_bins"]
        else:
            os.environ['BINS'] = config["key_third_bins"]
        key[l] = quant(key[l] - int(os.environ['BINS']) // 2 + 1, \
            max_tensors_k[l, :CHUNK_SIZE].cuda()).clone()
    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            os.environ['BINS'] = config["value_first_bins"]
        else:
            os.environ['BINS'] = config["value_second_bins"]
        value[l] = quant(value[l] - (int(os.environ['BINS']) // 2- 1), \
            max_tensors_v[l, :CHUNK_SIZE].clone().cuda()).clone()   
    return key, value 
if __name__ == "__main__":
    key, value = decode_function(args.path_to_encoded_kv, args.quantization_config, \
        args.model_config, args.chunk_size)
    model_config = json.load(open(args.model_config, "r"))
    kv_tuple = transformer_kv_to_tuple(key, value, model_config["num_heads"], model_config["heads_dim"])
    model, tokenizer = load_model(
            args.model_id,
            device="cuda",
            num_gpus=1,
            max_gpu_memory=f"{20}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )
    with open(args.input_text, "r") as f:
        text = f.read()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids=input_ids, \
            past_key_values=kv_tuple, max_new_tokens=10)
    print(tokenizer.decode(output[0][-10:]))
    # pickle.dump(kv_tuple, open("data/tmp.pkl", "wb"))
    # breakpoint()