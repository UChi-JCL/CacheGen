from src.encoder.encoder import CacheGenEncoder
import torch
import pickle
import torchac
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
    concat_tensor = None
    for i in range(start, end):
        if concat_tensor is None:
            concat_tensor = dict1[i].unsqueeze(0)
        else:
            concat_tensor = torch.cat((concat_tensor, \
                dict1[i].unsqueeze(0)), dim=0)
    return concat_tensor

if __name__ == "__main__":
    test_kv = pickle.load(open("data/test_kv.pkl", "rb"))
    fp_k, fp_v = transform_tuple_to_tensors(test_kv)
    encoder = CacheGenEncoder(fp_k=fp_k, fp_v=fp_v)
    encoder.quantize()
    cdf = encoder.compute_cdf(start_layer=0, end_layer=5, is_key=True)
    encode_input = concat_dict(encoder.quantized_key, 0, 5) + int(32) // 2 -1 
    # # cdf = cdf.unsqueeze(2).repeat(1, 1, 4096, 1)
    # print(encode_input.shape, cdf.shape)
    pickle.dump(cdf, open("data/test_cdf.pkl", "wb"))
    cdf = pickle.load(open("data/test_cdf.pkl", "rb"))
    bitstreams = []
    for i in range(2000):
        bits = torchac.encode_float_cdf(cdf, \
            encode_input[:, i].to(torch.int16) )
        bitstreams.append(bits)
    pickle.dump(bitstreams, open("data/test_bits.pkl", "wb"))
    breakpoint()