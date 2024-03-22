import torch 
import pickle
import os
import torchac_cuda
import time
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

if __name__ == "__main__":
    cdf = pickle.load(open("data/test_cdf.pkl", "rb"))
    cdf = _renorm_cast_cdf_(cdf, 16)
    output = torch.zeros( (2000, 5 * 4096 )).cuda().to(torch.int32)
    bits = pickle.load(open("data/test_bits.pkl", "rb"))
    for i in range(20):
        out = torchac_cuda.decode(output, cdf.unsqueeze(0), bits,  2000, 20, 100)
        print(out.reshape((2000, 5, 4096))[:, 2, 0].unique(return_counts=True))