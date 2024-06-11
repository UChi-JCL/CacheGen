from src.encoder.encoder import CacheGenEncoder
import torch
import pickle
import mytorchac
import json
from tqdm import tqdm
import numpy as np
import argparse
import time

p = argparse.ArgumentParser()
p.add_argument("--path_to_kv", 
               type = str, 
               default = "data_mistral_instruct_v0.2/test_kv_2.pkl", 
               help="Path to original KV cache")
p.add_argument("--quantization_config", 
               type = str, 
               default = "config/quantization_7b.json", 
               help="Path to quantization config")
p.add_argument("--chunk_size",
                type = int,
                default = 2000,
                help="Chunk size to encode")
p.add_argument("--output_path",
                type = str,
                default = "encoded/",
                help="Output path")
p.add_argument("--num_chunks",
                type = int,
                default = 4,
                help="Number of chunks")

args = p.parse_args()

def transform_tuple_to_tensors(kv):
    """ Takes in a tuple of #layers tuples, each of the tuple 
    is (key_tensor, value_tensor) and returns two tensors of shape
    (# layers, # tokens, # heads * head_dim)
    """
    head_num = kv[0][0].shape[1]
    head_dim = kv[0][0].shape[3]
    tokens_num = kv[0][0].shape[2]
    k_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
    v_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
    for i in range(len(kv)):
        k_tensor[i] = kv[i][0].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
        v_tensor[i] = kv[i][1].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
    return k_tensor, v_tensor

def _renorm_cast_cdf_(cdf, precision):
    """ The cdf normalization function in torchac
    """
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

def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)

def concat_dict(dict1, start, end):
    concat_tensor = None
    for i in range(start, end):
        if concat_tensor is None:
            concat_tensor = dict1[i].unsqueeze(0)
        else:
            concat_tensor = torch.cat((concat_tensor, \
                dict1[i].unsqueeze(0)), dim=0)
    return concat_tensor

def encode_function(path_to_original_kv, quantization_config, CHUNK_SIZE, output_path, c):
    """
    Given the path to the original key value cache, encode the KV cache
    Fields:
    - path_to_original_kv: the path to the original key value cache (Tuples)
    - quantization_config: the path to the quantization config
    - CHUNK_SIZE: the chunk size to encode, NEEDS to be multiples of 20!!!
    - output_path: the path to the output file
    """
    output_dict = {}
    config = json.load(open(quantization_config, "r"))
    test_kv = pickle.load(open(path_to_original_kv, "rb"))
    X = test_kv
    fp_k, fp_v = transform_tuple_to_tensors(X)
    l = fp_k.shape[0]
    encoder = CacheGenEncoder(fp_k=fp_k, fp_v=fp_v, config=config)
    encoder.quantize()
    cdf_k = encoder.compute_cdf(is_key=True)
    encode_input_key = concat_dict(encoder.quantized_key, 0, config["key_first_layers"])
    encode_input_key = torch.cat((encode_input_key, 
                                 concat_dict(encoder.quantized_key, config["key_first_layers"], config["key_second_layers"]) ), 
                                 dim=0)
    encode_input_key = torch.cat((encode_input_key, 
                                    concat_dict(encoder.quantized_key, config["key_second_layers"], l) ), 
                                 dim=0)
    
    cdf_v = encoder.compute_cdf(is_key=False)
    encode_input_value = concat_dict(encoder.quantized_value, 0, config["value_first_layers"])
    encode_input_value = torch.cat((encode_input_value, concat_dict(encoder.quantized_value, config["value_first_layers"], l) ), dim=0)
    cdf = torch.cat((cdf_k, cdf_v), dim=0)
    encode_input = torch.cat((encode_input_key, encode_input_value), dim=0)
    bitstreams = b""
    maxsize = 1024 * 160 * CHUNK_SIZE
    encode_function.BUFFER = np.zeros(maxsize, dtype=np.uint8)
    buffer = encode_function.BUFFER
    current_index = 0
    start_indices = []
    
    import pdb; pdb.set_trace()
    checkpoint1 = time.time()
    combined_bits = mytorchac.encode_float_cdf(cdf[0:1].repeat(50, 1, 1), 
                                      encode_input[0:1, 0:50].to(torch.int16).squeeze(0), 
                                      use_cuda=True, 
                                      max_out_size=1000,
                                      blockNum=1,
                                      threadNum=50)
    checkpoint2 = time.time()
    print(f"time is: {checkpoint2 - checkpoint1}")
    import pdb; pdb.set_trace()
    
    bits_sum = 0
    all_bits = []
    checkpoint1 = time.time()
    for k in range(0, 50):
        bits = mytorchac.encode_float_cdf(cdf[0:1], 
                                          encode_input[0:1, k].to(torch.int16), 
                                          use_cuda=False)
        bits_sum += len(bits)
        all_bits.append(bits)
    checkpoint2 = time.time()
    print(f"time is: {checkpoint2 - checkpoint1}")
    import pdb; pdb.set_trace()
    
    for l in range(cdf.shape[0]):
        checkpoint1 = time.time()
        print("Done with layer", l)
        import pdb; pdb.set_trace()
        for i in range(CHUNK_SIZE * c, CHUNK_SIZE * (c + 1)):
            # bits = mytorchac.encode_float_cdf(cdf[l:l+1], \
            #     encode_input[l:l+1, i].to(torch.int16) )
            bits = mytorchac.encode_float_cdf(cdf[l:l+1], 
                                              encode_input[l:l+1, i].to(torch.int16), 
                                              use_cuda=True, 
                                              max_out_size=1000,
                                              blockNum=1,
                                              threadNum=1)
            length = len(bits)
            start_indices += [current_index]
            buffer[current_index:current_index + length] = np.frombuffer(bits, dtype=np.uint8)
            current_index += length
        checkpoint2 = time.time()
        print(f"time is: {checkpoint2 - checkpoint1}")
    output_dict[f"bitstreams"] = torch.ByteTensor(list(buffer[:current_index].tobytes()))
    output_dict[f"start_indices"] =  torch.tensor(start_indices).int()
    output_dict["cdf"] = _renorm_cast_cdf_(cdf.float(), 16)
    output_dict["max_tensors_key"] = concat_max(encoder.max_tensors_key)
    output_dict["max_tensors_value"] = concat_max(encoder.max_tensors_value)
    pickle.dump(output_dict, open(output_path, "wb"))

if __name__ == "__main__":
    
    for i in tqdm(list(range(args.num_chunks))):
        encode_function(
            args.path_to_kv,
            args.quantization_config,
            args.chunk_size,
            f"{args.output_path}/{i}.pkl",
            i
        )