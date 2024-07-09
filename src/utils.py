import torch
from fastchat.model import load_model

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tuples], dim=0)

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
    new_kv = []
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]

        key_q, _ = torch_quant(bins, key)
        value_q, _ = torch_quant(bins, value)
        new_kv.append((key_q.unsqueeze(0)[:, :, :-1], value_q.unsqueeze(0)[:, :, :-1]))
    return tuple(new_kv)
