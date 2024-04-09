import torch
import time
import threading
import pickle
load_cache_stream = torch.cuda.Stream()
import torchac_cuda_layer
output = torch.zeros( (1 * 1000 * 160 * 1024 )).cuda().to(torch.int)
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
def decode_function(encoded_file, CHUNK_SIZE):
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
    cdf = encoded_file["cdf"]
    cdf = _renorm_cast_cdf_(cdf.float(), 16)
    
        
    # mapping  = np.zeros((CHUNK_SIZE, cdf.shape[0]*cdf.shape[1])) # num_input x N_symb
    # for i in range(len(mapping)):
    #     mapping[i] = np.arange(0, cdf.shape[0]*cdf.shape[1]) 
    # mapping = list(mapping.reshape(-1))
    # mapping = [int(x) for x in mapping]
    print("start decoding")
    nlayers = 160
    scale = 1
    
    for i in range(1): # 2 times to warm up the cache"
        bits = encoded_file["bitstreams"]
        concated_string = bits
        start_indices= encoded_file["start_indices"]
        print(output.shape, len(start_indices))
        # torchac_cuda.decode_fast(output, cdf.unsqueeze(0), concated_string.tobytes(),  \
        #     start_indices, CHUNK_SIZE, 20, CHUNK_SIZE//20)
        torchac_cuda_layer.decode_fast(output, cdf.unsqueeze(0), concated_string.tobytes(),  \
            start_indices, CHUNK_SIZE, nlayers*scale, CHUNK_SIZE, scale)
        
        # torchac.decode_float_cdf(cdf, concated_string[:start_indices[1]])
        # out = torchac_cuda.decode(output, cdf.unsqueeze(0), bits,  6000, 60, 100)
    #output.reshape((64, 100, 1024))
    print(output)
    # return None, None
    # out = output.reshape((2, max_tensors_k.shape[0], scale * CHUNK_SIZE, \
    #     model_config["hidden_dim"]))
def disk_load_worker(path, cpu_buf, dst):
    with torch.cuda.stream(load_cache_stream):
        x = torch.load(path)
        cpu_buf.copy_(x)
        dst.copy_(cpu_buf, non_blocking=True)
    print("load finish")

def disk_load_worker_naive(path, cpu_buf, dst):
    x = torch.load(path)
    cpu_buf.copy_(x)
    dst.copy_(cpu_buf)
    print("load finish")
    
#The following compute and load cannot be overlapped

x = torch.zeros((65247393))
xx = torch.rand((13000,13000)).cuda()
y = torch.empty((65247393)).cuda()
path = "x.pt"
# torch.save(x,path)
x = x.pin_memory()
print(x.device)
time.sleep(2)
st = time.monotonic()
# print(X[0])
for i in range(10):
    yy = torch.matmul(xx,xx)
torch.cuda.synchronize()
print("comp time", time.monotonic() - st )
X = pickle.load(open("data/test_encoded.pkl", "rb"))
for l in range(10):
    torch.cuda.synchronize()
    st = time.monotonic()
    t1 = threading.Thread(target=disk_load_worker, args=(path,x,y))
    #t1 = threading.Thread(target=disk_load_worker_naive, args=(path,x,y))
    t1.start()
    # yy = torch.matmul(xx,xx)
    decode_function(X, 1000)
   
    t1.join()
    torch.cuda.synchronize() #sync + join
    temp_time = time.monotonic() - st
    # end.record()
    # torch.cuda.synchronize()
    # temp_time = start.elapsed_time(end)
    print("total time", temp_time)