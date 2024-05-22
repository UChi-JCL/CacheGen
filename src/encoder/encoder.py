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
        
    def quantize(self):
        """ Quantize the key and value tensors 
        (self.fp_k and self.fp_v) 
        """
        for layer in range(len(self.fp_k)):
            
            if layer < self.config["key_first_layers"]:
                os.environ['BINS'] = self.config["key_first_bins"]
                
            elif layer < self.config["key_second_layers"]:
                os.environ['BINS'] = self.config["key_second_bins"]
            else:
                os.environ['BINS'] = self.config["key_third_bins"]
            tmp = torch_quant(self.fp_k[layer].float())
            self.quantized_key[layer] = tmp[0] + int(os.environ['BINS']) // 2 - 1
            self.max_tensors_key[layer] = tmp[1]
        for layer in range(len(self.fp_v)):
            if layer < self.config["value_first_layers"]:
                os.environ['BINS'] = self.config["value_first_bins"]
            else:
                os.environ['BINS'] = self.config["value_second_bins"]
            tmp = torch_quant(self.fp_v[layer].float())
            self.quantized_value[layer] = tmp[0]+ int(os.environ['BINS']) // 2 - 1
            self.max_tensors_value[layer] = tmp[1]
            
    def compute_cdf(self, is_key):
        """
        Compute the CDF based on the quantized tensors
        Field: 
        - start_layer: the start layer to compute the CDF
        - end_layer: the end layer to compute the CDF
        """
        # TODO: Add start_index here
        channels = self.fp_k[0].shape[-1]
        tokens = self.fp_k[0].shape[0]
        
        def process_batch(X, max_val):
            """
            input shape should be 【channels, tokens】
            """
            nchannels, ntokens = X.shape
            one_hot = torch.nn.functional.one_hot(X.long(), num_classes=max_val + 1).to(torch.float32)  # Use float32 to avoid integer overflow
            counts = one_hot.sum(dim=1) / ntokens
            ret = torch.cumsum(counts, dim=1).roll(1)
            ret[:, 0] = 0
            return ret

        def process_layers(X, max_val):
            """
            x is a iterator of dict values
            each element's shape is [tokens, channels]
            """
            results = []
            for x in X:
                ''' do permute here '''
                batch_counts = process_batch(x.cuda().permute(1, 0), max_val)
                results.append(batch_counts)

            final_counts = torch.cat(results, dim=0)
            
            return final_counts
        
        if is_key:
            X = self.quantized_key.values()
        else:
            X = self.quantized_value.values()
        value_range = 32
        cdfs = process_layers(X, value_range) # 4096 is batch size, ==> 18GB GPU memory
        final_cdf = cdfs.reshape((len(self.fp_k), channels, value_range+1)).cpu()
                
                
        return final_cdf