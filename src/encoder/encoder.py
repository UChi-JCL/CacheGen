import os
import torch
def torch_quant(qA):
    # shape (8, 2048)
    MAX = int(os.environ["BINS"]) // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float32)
    
    return xq, max1
class CacheGenEncoder:
    def __init__(self, **kwargs) -> None:
        """ 
        Fields: 
        - fp_kv: should be a tensor of shape (num_layers, num_tokens, num_channels)
        - fp_v: should be a tensor of shape (num_layers, num_tokens, num_channels)
        """
        self.fp_k = kwargs["fp_k"]
        self.fp_v = kwargs["fp_v"]
        
        self.quantized_key = {}
        self.max_tensors_key = {}  
        self.quantized_value = {}
        self.max_tensors_value = {} 
        self.config = kwargs["config"]
        
    def quantize(self, config):
        """ Quantize the key and value tensors 
        (self.fp_k and self.fp_v) 
        """
        for layer in range(len(self.fp_k)):
            
            if layer < self.config["key_first_layers"]:
                os.environ['BINS'] = self.config["key_first_bins"]
                
            elif layer < config["key_second_layers"]:
                os.environ['BINS'] = self.config["key_second_bins"]
            else:
                os.environ['BINS'] = self.config["key_third_bins"]
            tmp = torch_quant(self.fp_k[layer].float())
            self.quantized_key[layer] = tmp[0] + int(os.environ['BINS']) // 2 - 1
            self.max_tensors_key[layer] = tmp[1]
        for layer in range(len(self.fp_v)):
            if layer < config["value_first_layers"]:
                os.environ['BINS'] = self.config["value_first_bins"]
            else:
                os.environ['BINS'] = self.config["value_second_bins"]
            tmp = torch_quant(self.fp_v[layer].float())
            self.quantized_value[layer] = tmp[0]+ int(os.environ['BINS']) // 2 - 1
            self.max_tensors_value[layer] = tmp[1]
            
    def compute_cdf(self, is_key, config):
        """
        Compute the CDF based on the quantized tensors
        Field: 
        - start_layer: the start layer to compute the CDF
        - end_layer: the end layer to compute the CDF
        """
        # TODO: Add start_index here
        channels = self.fp_k[0].shape[-1]
        tokens = self.fp_k[0].shape[0]
        # if is_key:
        #     if end_layer <= config["key_first_layers"]:
        #         os.environ['BINS'] = self.config["key_first_bins"]
        #     elif end_layer <= config["key_second_layers"]:
        #         os.environ['BINS'] = self.config["key_second_bins"]
        #     else:
        #         os.environ['BINS'] = self.config["key_third_bins"]
        # else:
        #     if end_layer <= config["value_first_layers"]:
        #         os.environ['BINS'] = self.config["value_first_bins"]
        #     else:
        #         os.environ['BINS'] = self.config["value_second_bins"]
        final_cdf = torch.ones(len(self.fp_k), channels, 33)
        
        for i in range(len(self.fp_k)):
            # print("layer", i)
            if is_key:
                if i < config["key_first_layers"]:
                    os.environ['BINS'] = self.config["key_first_bins"]
                elif i < config["key_second_layers"]:
                    os.environ['BINS'] = self.config["key_second_bins"]
                else:
                    os.environ['BINS'] = self.config["key_third_bins"]
            else:
                if i < config["value_first_layers"]:
                    os.environ['BINS'] = self.config["value_first_bins"]
                else:
                    os.environ['BINS'] = self.config["value_second_bins"]
            for j in range(channels):
                
                if is_key:
                    tmp_input = self.quantized_key[i ][:, j]
                else:
                    tmp_input = self.quantized_value[i][:, j]
                # if end_layer == 40:
                #     breakpoint()
                symbs_orig, unique_tensor = torch.tensor(tmp_input).unique(return_counts=True)
                output_cdf = torch.zeros(32)
                output_cdf[symbs_orig.long()] = unique_tensor.float()
                output = output_cdf / output_cdf.sum()
                output = torch.cumsum(output_cdf ,dim=-1) / max(torch.cumsum(output_cdf , dim=-1))
                output =  torch.cat((torch.tensor([0.0]), output))
                final_cdf[i, j, :] = output
                
                
        return final_cdf