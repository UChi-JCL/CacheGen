from src.encoder.encoder import CacheGenEncoder
import torch
import pickle
import torchac
import json
def transform_tuple_to_tensors(kv):
    
    head_num = kv[0][0].shape[1]
    head_dim = kv[0][0].shape[3]
    tokens_num = kv[0][0].shape[2]
    k_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
    v_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
    for i in range(len(kv)):
        k_tensor[i] = kv[i][0].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
        v_tensor[i] = kv[i][1].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
    return k_tensor, v_tensor

def concat_dict(dict1, start, end):
    """ Concat the dict of CDF into a single tensor, start is 
    the start_layher, end is the end_layer
    """
    concat_tensor = None
    for i in range(start, end):
        if concat_tensor is None:
            concat_tensor = dict1[i].unsqueeze(0)
        else:
            concat_tensor = torch.cat((concat_tensor, \
                dict1[i].unsqueeze(0)), dim=0)
    return concat_tensor
def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)

def encode_function(path_to_original_kv, quantization_config, CHUNK_SIZE, output_path):
    """
    Given the path to the original key value cache, encode the KV cache
    Fields:
    - path_to_original_kv: the path to the original key value cache
    - quantization_config: the path to the quantization config
    - CHUNK_SIZE: the chunk size to encode, NEEDS to be multiples of 20!!! 
    - output_path: the path to the output file
    
    """
    output_dict = {}
    config = json.load(open(quantization_config, "r"))
    test_kv = pickle.load(open(path_to_original_kv, "rb"))
    fp_k, fp_v = transform_tuple_to_tensors(test_kv)
    l = fp_k.shape[0]
    encoder = CacheGenEncoder(fp_k=fp_k, fp_v=fp_v, config=config)
    encoder.quantize(config=config)
    cdf_k = encoder.compute_cdf(is_key=True, config=config)
    encode_input_key = concat_dict(encoder.quantized_key, 0, config["key_first_layers"]) 
    encode_input_key = torch.cat((encode_input_key, \
        concat_dict(encoder.quantized_key, config["key_first_layers"], config["key_second_layers"]) ), dim=0)
    encode_input_key = torch.cat((encode_input_key, \
        concat_dict(encoder.quantized_key, config["key_second_layers"], l) ), dim=0)
    
    cdf_v = encoder.compute_cdf(is_key=False, config=config)
    encode_input_value = concat_dict(encoder.quantized_value, 0, config["value_first_layers"])
    encode_input_value = torch.cat((encode_input_value, \
        concat_dict(encoder.quantized_value, config["value_first_layers"], l) ), dim=0)
    cdf = torch.cat((cdf_k, cdf_v), dim=0)
    encode_input = torch.cat((encode_input_key, encode_input_value), dim=0)
    
    # # cdf = cdf.unsqueeze(2).repeat(1, 1, 4096, 1)
    # print(encode_input.shape, cdf.shape)
    bitstreams = b""
    start_indices = []  
    for i in range(CHUNK_SIZE):
        if i % 100 == 0:
            print(i)
        bits = torchac.encode_float_cdf(cdf, \
            encode_input[:, i].to(torch.int16) )
        # bitstreams.append(bits)
        start_indices += [len(bitstreams)]
        bitstreams += bits
    output_dict["cdf"] = cdf
    output_dict["bitstreams"] = bitstreams
    output_dict["start_indices"] = start_indices
    output_dict["max_tensors_key"] = concat_max(encoder.max_tensors_key)
    output_dict["max_tensors_value"] = concat_max(encoder.max_tensors_value)
    pickle.dump(output_dict, open(output_path, "wb"))
    
if __name__ == "__main__":
    encode_function("data/test_kv.pkl", "config/quantization_7b.json", 2000, "data/test_encoded.pkl")