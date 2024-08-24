from src.utils import *
import numpy as np
import argparse 
p = argparse.ArgumentParser()
p.add_argument("--num_chunks", type=int, default=10)
p.add_argument("--output_path", type=str)
p.add_argument("--num_data", type=int, default = 10) 
args= p.parse_args()

if __name__ == "__main__":
    f = open(args.output_path, "w")
    for i in range(args.num_data):
        np.random.seed(i)
        all_bws = bw_generator(args.num_chunks)
        json.dump({"bw": list(all_bws)}, f)
        f.write("\n")
    f.close()