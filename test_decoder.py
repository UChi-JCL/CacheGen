import torch 
import pickle
import os
import torchac_cuda
import time
import json
import argparse
import numpy as np
import torchac_cuda_layer
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
        
        tmp1 = key[i].reshape((key[i].shape[0], block, thread)).permute((1, 0, 2)).unsqueeze(0).cuda()
        tmp2 = value[i].reshape((key[i].shape[0], block, thread)).permute((1, 0, 2)).unsqueeze(0).cuda()
        kv_list += [(tmp1, tmp2)]
    return tuple(kv_list)
def merge_kv(left, right, free_left = False, free_right = False):
    """
    Merges two kv caches, returns a merged KV cache
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - left: the left kv cache, could be None
    - right: the right kv cache

    Returns: The merged kv cache. If left is None, returns right
    """
    if left is None:
        return right
    #assert len(left) == len(right)

    def generator():
        for left_layer, right_layer in zip(left, right):
            yield (torch.cat([left_layer[0], right_layer[0]], dim = -2), torch.cat([left_layer[1], right_layer[1]], dim = -2))
            if free_left:
                del left_layer
            if free_right:
                del right_layer

    return tuple(generator())

def decode_function(chunk_id, path_to_encoded_kv, quantization_config, model_config, CHUNK_SIZE):
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
    encoded_file = pickle.load(open(path_to_encoded_kv + f"_chunk_{chunk_id}.pkl", "rb"))
    cdf = encoded_file["cdf"]
    cdf = _renorm_cast_cdf_(cdf.float(), 16)
    
    
    max_tensors_k = encoded_file["max_tensors_key"]
    max_tensors_v = encoded_file["max_tensors_value"]
    # mapping  = np.zeros((CHUNK_SIZE, cdf.shape[0]*cdf.shape[1])) # num_input x N_symb
    # for i in range(len(mapping)):
    #     mapping[i] = np.arange(0, cdf.shape[0]*cdf.shape[1]) 
    # mapping = list(mapping.reshape(-1))
    # mapping = [int(x) for x in mapping]
    print("start decoding")
    nlayers = cdf.shape[0]
    scale = 1
    
    for i in range(1): # 2 times to warm up the cache"
        encoded_file = pickle.load(open(path_to_encoded_kv + f"_chunk_{chunk_id}.pkl", "rb"))
        bits = encoded_file["bitstreams"]
        output = torch.zeros( (scale * CHUNK_SIZE * cdf.shape[0] * cdf.shape[1] )).cuda().to(torch.int32)
        start_indices= torch.tensor(encoded_file["start_indices"])
        input_bitstreams = torch.ByteTensor(list(bits.tobytes()))
        st = time.monotonic()
        start_indices = start_indices.pin_memory()
        start_indices = start_indices.cuda().int()
        
        # input_bitstreams = bits.tobytes()
            # print(output.shape, len(start_indices))
        # input_bitstreams = torch.cat((input_bitstreams, torch.zeros((10000)).to(torch.uint8).cuda() ), dim=0)
        
        # torchac_cuda.decode_fast(output, cdf.unsqueeze(0), concated_string.tobytes(),  \
        #     start_indices, CHUNK_SIZE, 20, CHUNK_SIZE//20)
        torchac_cuda.decode_fast(output, cdf.unsqueeze(0), input_bitstreams,  \
            start_indices, CHUNK_SIZE, nlayers*scale, CHUNK_SIZE, scale)
        torch.cuda.synchronize()
        # torchac.decode_float_cdf(cdf, concated_string[:start_indices[1]])
        # out = torchac_cuda.decode(output, cdf.unsqueeze(0), bits,  6000, 60, 100)
        print( f"TTFT: {time.monotonic() - st}")
        del input_bitstreams
    #output.reshape((64, 100, 1024))
    print(output)
    # breakpoint()
    # return None, None
    out = output.reshape((2, max_tensors_k.shape[0], scale * CHUNK_SIZE, \
        model_config["hidden_dim"]))
    key = out[0].half()
    value = out[1].half()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            os.environ['BINS'] = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            os.environ['BINS'] = config["key_second_bins"]
        else:
            os.environ['BINS'] = config["key_third_bins"]
        # breakpoint()
        # key[l] = quant(key[l] - int(os.environ['BINS']) // 2 + 1, \
        #     max_tensors_k[l, :scale * CHUNK_SIZE]).clone()
        key[l] = quant(key[l] - int(os.environ['BINS']) // 2 + 1, \
            max_tensors_k[l, scale*CHUNK_SIZE*c:(c+1)*scale*CHUNK_SIZE].cuda()).clone()
    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            os.environ['BINS'] = config["value_first_bins"]
        else:
            os.environ['BINS'] = config["value_second_bins"]
        value[l] = quant(value[l] - (int(os.environ['BINS']) // 2- 1), \
            max_tensors_v[l, scale*CHUNK_SIZE*c:(c+1)*scale*CHUNK_SIZE].clone().cuda()).clone()   
        # value[l] = quant(value[l] - (int(os.environ['BINS']) // 2- 1), \
        #     max_tensors_v[l, :scale * CHUNK_SIZE].clone()).clone()   
    return key, value 
if __name__ == "__main__":
    
    model_config = json.load(open(args.model_config, "r"))
    kv_tuple = None
    for c in range(5):
        key, value = decode_function(c, args.path_to_encoded_kv, args.quantization_config, \
            args.model_config, args.chunk_size)
        chunk_kv = transformer_kv_to_tuple(key, value, model_config["num_heads"], model_config["heads_dim"])
        kv_tuple = merge_kv(kv_tuple, chunk_kv)
    # pickle.dump(kv_tuple, open("data/tmp.pkl", "wb"))
    # kv_tuple = pickle.load(open("data/tmp.pkl", "rb"))
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
            past_key_values=kv_tuple, max_new_tokens=20)
    print(tokenizer.decode(output[0][-20:]))
    # pickle.dump(kv_tuple, open("data/tmp.pkl", "wb"))
