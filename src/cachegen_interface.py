from typing import Tuple, TypeAlias, Union, List
from dataclasses import dataclass
import pickle
import hashlib
import torch
import os
import torchac_cuda
import torchac
import time
from src.server.client import *

from src.encoder.encoder import CacheGenEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
KVCache: TypeAlias = Tuple[Tuple[torch.Tensor]]
CHUNK_SIZE = 6000
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


def vectorwise_quant(xq, max1, dim=-1, quant_type="vector"):
    
    C = int(os.environ["BINS"]) // 2 - 1
    x = (xq / C * max1).to(torch.float16)
    return x

@dataclass
class CacheGenConfig:
    """
    The configuration of how to do prefill in cache gen
    Fields:
    - start_index, end_index: indicating the data (either kv cache or tokenized input) corresponds to 
                              [start_index, end_index) of the whole tokenized input (i.e., input_ids[start_index:end_index])
    - is_kv: a boolean value, indicating if the underlying data is kv cache or tokenized input (input ids)
    - data: can be either kv cache (usually is a Tuple[Tuple[torch.Tensor, torch.Tensor]]) or tokenized input (just torch.Tensor)

    NOTE: we do not support splitting on raw text data, because the mapping between text and tokenized text is not deterministic.
          For example, "我吃水果" might be tokenized to 3 tokens where "水果" is mapped to a single token. But if
          we split the raw text to ["我", "吃", "水", "果"], it will be mapped to 4 tokens.
    """

    ''' start index on the whole tokenized input, inclusive '''
    start_index: int

    ''' end index on the whole tokenized input, exclusive '''
    end_index: int

    ''' the flag indicating whether the data is kv cache or tokenized input '''
    is_kv: bool

    ''' the underlying data, could be either kv or tokenzied input '''
    data: Union[torch.Tensor, KVCache]

class CacheGenEngine:
    """
    The CacheGen engine
    - check if text/token list exists in cache
    - given text/token list, retrieve kv cache chunks
    - given text/token list and kv, store the kv into cachegen engine
    """

    def __init__(self, *args, **kwargs):
        self.model = kwargs['model']
        self.input_id_to_k = {}
        self.input_id_to_v = {} 
        l = int(os.environ['LAYERS'])
        c = int(os.environ['CHANNELS'])
        N = int(os.environ['TOKENS'])
        self.output_tensor = [torch.zeros((CHUNK_SIZE, l * c )).cuda().to(torch.int32) for i in range(2)]
        self.thread = int(os.environ['THREADS'])
        self.block = int(os.environ['BLOCKS'])
        print("Initializing the CDFs")
        self.k_cdf = torch.zeros((l, c, 33)).half()
        self.v_cdf = torch.zeros((l, c, 33)).half()
        self.l = l
        self.c = c
        self.N = N
        print("Done")
        self.max_tensors_k = {}
        self.max_tensors_v = {}
        self.config = kwargs['config']
        

    def input_ids_to_key(self, input_ids: torch.Tensor) -> str:
        """
        Compute the unique key based on the value of the input_ids
        """
        # TODO: this is a helper function, please implement your own logic here
        tensor_bytes = input_ids.cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()

    def contains(self, input_hash , chunk_id) -> bool:
        """
        Check if the input ids is in the cache gen engine

        Input:
            input_ids: a torch.Tensor whose shape of N elements. We assume that there is NO batching dimension

        Returns:
            True if the input ids is in the cache gen engine, False otherwise
        """
        # TODO: maybe do the "prefix check" here
        if (input_hash, chunk_id) in self.max_tensors_k:
            return True
        return False

    def get(self, input_ids: torch.Tensor) -> List[CacheGenConfig]:
        """
        Get the KV cache from the input ids
        Will be called by CacheGenController.get(). 
        This function includes fetching the KV cache from the storage server, 
        and decode 
        """

        #### Here are two examples
        #### Example 1: 
        #if key in self.cached_kv:
        #    print("\033[33mFound existing kv cache!\033[0m") 
        #    past_kv = self.cached_kv[key]
        #    # need to remove the last element
        #    final_loc = min(len(input_ids) - 1, past_kv[0][0].shape[-2])
        #    kv = split_kv(past_kv, final_loc)[0] 
        #    return [CacheGenConfig(0, len(input_ids) - 1, True, kv)]
        #else:
        #    print("\033[33mNo kv cache found, returing the text\033[0m") 
        #    return [CacheGenConfig(0, len(input_ids) - 1, False, None)]

        #### Example 2:
        # loc1, loc2 = 1800, 2000 
        # final_loc = min(len(input_ids) - 1, self.input_id_to_kv[input_ids_hash][0][0].shape[-2])
        # kv1, kv2 = split_kv(self.input_id_to_kv[input_ids_hash], loc1)
        # _, kv2 = split_kv(self.input_id_to_kv[input_ids_hash], loc2)
        # return [CacheGenConfig(0,    loc1,      True,  kv1),
        #        CacheGenConfig(loc1, loc2,      True, kv2),
        #        CacheGenConfig(loc2, final_loc, False,  input_ids[loc2:final_loc])]

        # return []
        input_ids_hash = self.input_ids_to_key(input_ids[0])
        configs = []
        for i in range(0, self.N, CHUNK_SIZE):
            if self.contains(input_ids_hash, i//CHUNK_SIZE):
                # kv = pickle.load(open("data/test_kv.pkl", "rb") )
                # kv_tuple = split_kv(kv, CHUNK_SIZE)[0]
                # del kv
                target_latency = 0.2
                # encoded_keys = send_request(0, 0, 0, 0, target_latency)  
                encoded_keys = self.input_id_to_k[(input_ids_hash, i//CHUNK_SIZE)]
                cdf_key = self.k_cdf
                target_latency = 0.2
                # encoded_values = send_request(0, 0, 0, 1, target_latency)
                encoded_values = self.input_id_to_v[(input_ids_hash, i//CHUNK_SIZE)]
                cdf_value = self.v_cdf
                # output_tensor = torch.zeros((CHUNK_SIZE, self.l * self.c )).cuda().to(torch.int32)
                st = time.monotonic()
                torchac_cuda.decode(self.output_tensor[0], cdf_key.unsqueeze(0), \
                    encoded_keys, CHUNK_SIZE, 20, CHUNK_SIZE//20)
                
                key = self.output_tensor[0].reshape((CHUNK_SIZE, self.l, self.c)).permute(1, 0, 2)
                del self.output_tensor[0]
                torchac_cuda.decode(self.output_tensor[0], cdf_value.unsqueeze(0), \
                    encoded_values, CHUNK_SIZE, 20, CHUNK_SIZE//20)
                print(f"Decoding time: {time.monotonic() - st}")
                value = self.output_tensor[0].reshape((CHUNK_SIZE, self.l, self.c)).permute(1, 0, 2)
                key = key.half()
                value = value.half()  
                # sanity =  torch.load("sanity.pt")    
                for l in range(key.shape[0]):
                    if l < self.config["key_first_layers"]:
                        os.environ['BINS'] = self.config["key_first_bins"]
                    elif l < self.config["key_second_layers"]:
                        os.environ['BINS'] = self.config["key_second_bins"]
                    else:
                        os.environ['BINS'] = self.config["key_third_bins"]
                    key[l] = vectorwise_quant(key[l] - int(os.environ['BINS']) // 2 + 1, \
                        self.max_tensors_k[(input_ids_hash, i)][l, i:i+CHUNK_SIZE].cuda()).clone()
                    
                    # key[l] = kv_tuple[l][0].permute((0, 2, 1, 3)).reshape((CHUNK_SIZE, self.block* self.thread))
                    
                for l in range(value.shape[0]):
                    if l < self.config["value_first_layers"]:
                        os.environ['BINS'] = self.config["value_first_bins"]
                    else:
                        os.environ['BINS'] = self.config["value_second_bins"]
                    value[l] = vectorwise_quant(value[l] - (int(os.environ['BINS']) // 2- 1), \
                        self.max_tensors_v[(input_ids_hash, i)][l, i:i+CHUNK_SIZE].clone().cuda()).clone()
                    # value[l] = kv_tuple[l][1].permute((0, 2, 1, 3)).reshape((CHUNK_SIZE, self.block* self.thread))
                kv_tuple = self.transformer_kv_to_tuple(key, value)
                configs.append(CacheGenConfig(i, i + CHUNK_SIZE, True, kv_tuple))
                del self.output_tensor
            else:
                configs.append(CacheGenConfig(i, min(self.N-1, i+CHUNK_SIZE) , False, input_ids[:, i:min(self.N-1, i+CHUNK_SIZE)]))
        return configs
    def transformer_kv_to_tuple(self, key, value):
        """ Field:
        - key: a torch.Tensor of shape (l, N, 4096)
        """
        kv_list = []
        for i in range(len(key)):
            
            tmp1 = key[i].reshape((key[i].shape[0], self.block, self.thread)).permute((1, 0, 2)).unsqueeze(0)
            tmp2 = value[i].reshape((key[i].shape[0], self.block, self.thread)).permute((1, 0, 2)).unsqueeze(0)
            kv_list += [(tmp1, tmp2)]
        return tuple(kv_list)

    def transform_tuple_to_tensors(self, kv):
        head_num = kv[0][0].shape[1]
        head_dim = kv[0][0].shape[3]
        tokens_num = kv[0][0].shape[2]
        k_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
        v_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
        for i in range(len(kv)):
            k_tensor[i] = kv[i][0].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
            v_tensor[i] = kv[i][1].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
        return k_tensor, v_tensor
    def concat_dict(self, dict1, start, end):
        concat_tensor = None
        for i in range(start, end):
            if concat_tensor is None:
                concat_tensor = dict1[i].unsqueeze(0)
            else:
                concat_tensor = torch.cat((concat_tensor, \
                    dict1[i].unsqueeze(0)), dim=0)
        return concat_tensor
    def cdf_helper(self, encoder, start_layer, end_layer, is_key, start_index=0, config=None):
        """ Field:
        - start_layer: the start layer to compute the CDF
        - end_layer: the end layer to compute the CDF
        - is_key: a boolean value, indicating if it's key or value
        """
        cdf = encoder.compute_cdf(start_layer=start_layer, end_layer=end_layer, \
            is_key=is_key, config=config)
        # TODO
        if is_key:
            self.k_cdf[start_layer:end_layer, :, :cdf.shape[-1]] = cdf
            if cdf.shape[-1] < 33:
                # Fill the later half to 1
                self.k_cdf[start_layer:end_layer, :, cdf.shape[-1]:] = 1
        else:
            self.v_cdf[start_layer:end_layer, :, :cdf.shape[-1]] = cdf
            if cdf.shape[-1] < 33:
                # Fill the later half to 1
                self.v_cdf[start_layer:end_layer, :, cdf.shape[-1]:] = 1
    def encoder_helper(self, encoder, is_key, start_index, config=None):
        bitstreams = []
        if is_key:
            cdfs = self.k_cdf
            # encode_input = self.concat_dict(encoder.quantized_key, 0, 3) 
            encode_input = self.concat_dict(encoder.quantized_key, 0, config["key_first_layers"]) 
            encode_input = torch.cat((encode_input, \
                self.concat_dict(encoder.quantized_key, config["key_first_layers"], config["key_second_layers"]) ), dim=0)
            encode_input = torch.cat((encode_input, \
                self.concat_dict(encoder.quantized_key, config["key_second_layers"], self.l) ), dim=0)
            
        else:
            cdfs = self.v_cdf
            encode_input = self.concat_dict(encoder.quantized_value, 0, config['value_first_layers']) 
            encode_input = torch.cat((encode_input, \
                self.concat_dict(encoder.quantized_value, config["value_first_layers"], self.l) ), dim=0) 
        print("Start encoding")
        for i in range(CHUNK_SIZE * start_index, CHUNK_SIZE * (start_index + 1)):
            bitstreams.append(torchac.encode_float_cdf(cdfs.float(), \
                encode_input[:, i].to(torch.int16) ))
            # output_tensor = torch.zeros((1, self.l * self.c )).cuda().to(torch.int32)
            # cdf_norm = _renorm_cast_cdf_(cdfs.float(), 16)
            # torchac_cuda.decode(output_tensor, \
            #     cdf_norm.unsqueeze(0), bitstreams, 1, 1, 1 )
        return bitstreams
    def concat_max(self, max1):
        """
        Given a dict of max tensors, concatenate them into a single tensor
        """
        maxes = []
        for i in range(len(max1)):
            maxes.append(max1[i].unsqueeze(0))
        return torch.cat(maxes, dim=0)
        
    def set(self, input_ids: torch.Tensor, kv: KVCache, quantization_config: dict, offline=True):
        """
        Set the model-generated KV cache for a given input ids
        input_ids: a torch.Tensor whose shape of (1, N) elements. We assume that there is NO batching dimension
        Note:
            It's possible that the input_ids is already in the cache gen engine. In this case, we should skip (or maybe overwrite?)
        """
        # Do encoding here 
        # TODO: replace with GPU-encoder here 
        input_id_hash = self.input_ids_to_key(input_ids)
        fp_k, fp_v = self.transform_tuple_to_tensors(kv)
        encoder = CacheGenEncoder(fp_k=fp_k, fp_v=fp_v, config=self.config)
        encoder.quantize(config=quantization_config)
        if offline:
            self.cdf_helper(encoder, 0, quantization_config["key_first_layers"], True, 0, quantization_config)
            self.cdf_helper(encoder, quantization_config["key_first_layers"], quantization_config["key_second_layers"], True, 0, quantization_config)
            self.cdf_helper(encoder, quantization_config["key_second_layers"], self.l, True, 0, quantization_config)
            
            self.cdf_helper(encoder, 0, quantization_config["value_first_layers"], False, 0, quantization_config)
            self.cdf_helper(encoder,  quantization_config["value_first_layers"], self.l, False, 0, quantization_config)
            pickle.dump(self.concat_max(encoder.max_tensors_key), open(f"{os.environ['TMP_DIR']}/test_max_k.pkl", "wb"))
            bitstreams = self.encoder_helper(encoder, True, 0, quantization_config)
            pickle.dump(self.concat_max(encoder.max_tensors_value), open(f"{os.environ['TMP_DIR']}/test_max_v.pkl", "wb"))
            pickle.dump(bitstreams, open(f"{os.environ['TMP_DIR']}/test_bits_k.pkl", "wb"))
            self.input_id_to_k[(input_id_hash, 0)] = bitstreams
            bitstreams = self.encoder_helper(encoder, False, 0, quantization_config)

            pickle.dump(bitstreams, open(f"{os.environ['TMP_DIR']}/test_bits_v.pkl", "wb"))
            pickle.dump(self.k_cdf, open(f"{os.environ['TMP_DIR']}/test_cdf_k.pkl", "wb"))
            pickle.dump(self.v_cdf, open(f"{os.environ['TMP_DIR']}/test_cdf_v.pkl", "wb"))
        else:
            self.k_cdf = pickle.load(open(f"{os.environ['TMP_DIR']}/test_cdf_k.pkl", "rb"))
            self.v_cdf = pickle.load(open(f"{os.environ['TMP_DIR']}/test_cdf_v.pkl", "rb"))
            
            self.k_cdf = _renorm_cast_cdf_(self.k_cdf.float(), 16)
            self.v_cdf = _renorm_cast_cdf_(self.v_cdf.float(), 16)
            self.input_id_to_k[(input_id_hash, 0)] = pickle.load(open(f"{os.environ['TMP_DIR']}/test_bits_k.pkl", "rb"))
            self.input_id_to_v[(input_id_hash, 0)] = pickle.load(open(f"{os.environ['TMP_DIR']}/test_bits_v.pkl", "rb"))
            self.max_tensors_k[(input_id_hash, 0)] = pickle.load(open(f"{os.environ['TMP_DIR']}/test_max_k.pkl", "rb"))
            self.max_tensors_v[(input_id_hash, 0)] = pickle.load(open(f"{os.environ['TMP_DIR']}/test_max_v.pkl", "rb"))
    def merge_kv(self, input_ids: torch.Tensor):
        """
        Perform inference on the input_ids
        """
        cachegen_configs = self.get(input_ids)
        merged_kv = None
        for config in cachegen_configs:
            if config.is_kv:
                merged_kv = merge_kv(merged_kv, config.data, free_left = True, free_right = True)
            else:
                st= time.monotonic()
                end_index = config.end_index
                generated = self.model.generate(inputs=input_ids[:, :end_index], 
                                                past_key_values=merged_kv,
                                                return_dict_in_generate=True,
                                                max_new_tokens = 1)
                del merged_kv
                merged_kv = generated.past_key_values
                print(f"Time for generating from scratch: {time.monotonic() - st}")
        return merged_kv


class CacheGenController:
    """
    The cachegen controller
    Currently designed as a singleton class, because we do not want to load the file multiple times. Calling
    `CacheGenController.GetInstance()` will get the singleton object. This behavior could be changed in the future.
    """
    _instances = {}

    def __init__(self, *args, **kwargs):
        self.engine = CacheGenEngine(*args, **kwargs)
        

    def get(self, input_ids: torch.Tensor) -> List[CacheGenConfig]:
        """
        Get the list of CacheGenConfig from the input ids
        Input:
            input_ids: a torch.Tensor whose shape of N elements. We assume that there is NO batching dimension

        Returns:
            A list of `CacheGenConfig` object, containing the kv cache or tokenized inputs.
            The elements in the list should be sorted by `start_index`.
            For any i, we should have `result[i].end_index == result[i+1].start_index`
            If result[-1] is a kv cache, we should make sure that result[-1].end_index == len(input_ids) - 1

        NOTE:
        - Remember to remove the batching dimension when calling this function
        - Make sure the returned list satisifies the requirements
        """
        return self.engine.get(input_ids)
    
    def set(self, input_ids: torch.Tensor, kv: KVCache, quantization_config: dict, offline=True):
        """
        Set the model-generated KV cache for a given input ids
        This interface is for potential CacheGen functionalities
        """
        self.engine.set(input_ids, kv, quantization_config, offline)  

    @classmethod
    def GetInstance(cls, *args, **kwargs):
        """
        Return the singleton instance of the CacheGen controller
        """
        if cls not in cls._instances:
            instance = cls(*args, **kwargs)

            cls._instances[cls] = instance

        return cls._instances[cls]

###########
# Helper functions
###########

def merge_kv(left: KVCache, right: KVCache, free_left = False, free_right = False) -> KVCache:
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

def split_kv(kv: KVCache, split_index: int) -> Tuple[KVCache, KVCache]:
    """
    Splits a kv cache into two kv caches
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - kv: the kv cache to be splitted
    - split_index: the index to split the kv cache

    Returns: a tuple of two kv caches
    """
    def generator_left():
        for layer in kv:
            yield (layer[0][:, :, :split_index], layer[1][:, :, :split_index])
    left = tuple(generator_left())
    def generator_right():
        for layer in kv:
            yield (layer[0][:, :, split_index:], layer[1][:, :, split_index:])
    right = tuple(generator_right())
    return left, right

if __name__ == "__main__":
    ## Test the merge_kv function
    def generate_kvcache(shape) -> KVCache:
        """
        Generates a kv cache with the following shape 
        """
        len = 32
        for i in range(len):
            yield (torch.randn(shape), torch.randn(shape))
    kv1 = tuple(generate_kvcache((1, 32, 500, 128)))
    kv2 = tuple(generate_kvcache((1, 32, 800, 128)))
    merged = merge_kv(kv1, kv2)
    print(merged[0][0].shape)
