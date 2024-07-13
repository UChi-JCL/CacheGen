import torch
from fastchat.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def define_model_and_tokenizer(model_id, num_gpus=1, max_gpu_memory=48):
    """ Define the model and tokenizer
    """
    model, tokenizer = load_model(
            model_id,
            device="cuda",
            num_gpus=num_gpus,
            max_gpu_memory=f"{max_gpu_memory}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )

    return model, tokenizer


def tensor_to_tuple(kv):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0), 
                       kv[i][1].unsqueeze(0)))
    return tuple(new_kv)

def torch_quant(bins: int, qA: torch.Tensor):
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float16)
    
    return x, max1

def default_quantization(kv, bins):
    """ Quantize the key value tensors into tuple of key and value tensors
    """
    channels = kv.shape[-1] * kv.shape[-3]
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)
        key, _ = torch_quant(bins, key)
        value, _ = torch_quant(bins, value)
        quant_key = key.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        quant_value = value.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        # quant_key, _ = torch_quant(bins, key.flatten())
        # quant_value, _ = torch_quant(bins, value.flatten())
        # quant_key = quant_key.reshape(key.shape)
        # quant_value = quant_value.reshape(value.shape)
        kv[i][0] = quant_key
        kv[i][1] = quant_value
    kv = kv[:, :, :, :-10]
    return tensor_to_tuple(kv)

