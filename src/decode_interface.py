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


def transformer_kv_to_tuple(key, value):
    """ Given the key and value tensors, returns a tuple of tuples
    Field:
    - key: a torch.Tensor of shape (#layers, #tokens, #channels)
    - value: a torch.Tensor of shape (#layers, #tokens, #channels)
    """
    kv_list = []
    for i in range(len(key)):
        
        tmp1 = key[i].permute((1, 0, 2)).unsqueeze(0)
        tmp2 = value[i].permute((1, 0, 2)).unsqueeze(0)
        kv_list += [(tmp1, tmp2)]
    return tuple(kv_list)
def quant(xq, max1, dim=-1, quant_type="vector"):
    
    C = int(os.environ["BINS"]) // 2 - 1
    x = (xq / C * max1).to(torch.float16)
    return x


def decode_function(cdf, bits, start_indices, max_tensors_k, max_tensors_v, quantization_config, CHUNK_SIZE, output, chunk_id):
    """
    Given the path to the encoded key value cache, decode the KV cache
    Fields:
    - cdf: the cdf tensor (used to encode/decode the KV)
    
    - path_to_encoded_kv: the path to the encoded key value cache
    - quantization_config: the path to the quantization config
    - model_config: the path to the model config
    - CHUNK_SIZE: the chunk size to decode, NEEDS to be multiples of 20!!! 
    Outputs:
    - key: the decoded key tensor in the shape of (layers, num_heads, tokens, heads_dim)
    """
    config = quantization_config
    start_time = time.monotonic()
    concated_string = bits
    nlayers = cdf.shape[0]
    scale = 2
    kernel_start = time.monotonic()
    start_indices = torch.tensor(start_indices).int().cuda()
    torchac_cuda.decode_fast(output, 
                            cdf.unsqueeze(0), 
                            concated_string, 
                            start_indices, 
                            CHUNK_SIZE, 
                            nlayers * scale, 
                            1000, 
                            scale)
    print("kernel computation time: ", time.monotonic() - kernel_start)
    out = output.reshape((2, max_tensors_k.shape[0], CHUNK_SIZE, 1024))
    key = out[0].half()
    value = out[1].half()
    max_tensors_k = max_tensors_k.cuda()
    max_tensors_v = max_tensors_v.cuda()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            os.environ['BINS'] = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            os.environ['BINS'] = config["key_second_bins"]
        else:
            os.environ['BINS'] = config["key_third_bins"]
        key[l] = quant(key[l] - int(os.environ['BINS']) // 2 - 1, max_tensors_k[l, chunk_id * CHUNK_SIZE: (chunk_id + 1) * CHUNK_SIZE])

    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            os.environ['BINS'] = config["value_first_bins"]
        else:
            os.environ['BINS'] = config["value_second_bins"]
        value[l] = quant(value[l] - (int(os.environ['BINS']) // 2 - 1), max_tensors_v[l, chunk_id * CHUNK_SIZE: (chunk_id + 1) * CHUNK_SIZE])
    key = key.reshape(
        key.shape[0],
        key.shape[1],
        8,
        128)
    value = value.reshape(
        value.shape[0],
        value.shape[1],
        8,
        128)
    kv_tuple = transformer_kv_to_tuple(key, value)
    torch.cuda.synchronize()
    print("per iteration total time: ", time.monotonic() - start_time)
    return kv_tuple
