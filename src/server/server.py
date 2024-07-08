from http.server import BaseHTTPRequestHandler, HTTPServer
import asyncio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import pickle
import time
import logging
import os
import torch
import argparse
p = argparse.ArgumentParser()
p.add_argument("--encoded_dir", type = str, default = "encoded", help="The directory to encoded kv. ")
args = p.parse_args()
logging.basicConfig(format= '[%(asctime)s] %(levelname)s - %(message)s', level=logging.INFO)

def get_sleep_time(chunk_size_bytes, rate_kBytesPerSec, speed_up_ratio=1.0):
    """
    Returns the sleep time for the given chunk size and rate
    Speed up ratio is used to speed up the sending process
    """
    return chunk_size_bytes / 1024 / rate_kBytesPerSec / speed_up_ratio

class BitstreamServer:
    """
    Initialize once and serves the file request
    """

    def __init__(self):
        self.files_dict = {}
        all_files = os.listdir(args.encoded_dir)
        # all_docs = []
        quality = 0
        # for file in all_files:
        #     doc_id = int(file.split("doc_")[1].split("_")[0])
        #     all_docs.append(doc_id)
        
        for file in all_files:
            if ".pkl" not in file: continue
            doc_id = int(file.split("doc_")[1].split("_")[0])
            chunk_id = int(file.split("doc_")[1].split("_")[-1].split(".")[0])
            file_key = (doc_id, chunk_id, quality)
            if file_key not in self.files_dict:
                self.files_dict[file_key] = pickle.load(open(f"{args.encoded_dir}/{file}", "rb")) # torch.load(f"{args.encoded_dir}/{file}")
        # self.files_dict[()]
        print("Data loaded successfully, ", self.files_dict.keys())
    def get(self, document_id, chunk_id, quality):
        """
        Returns the bytestream for the given parameters
        """
        file_key = (document_id, chunk_id, quality)
        print("File key is: ", file_key)
        if file_key not in self.files_dict:
            logging.warning("Data not found for the given parameters")
            return None

        file_data = self.files_dict.get(file_key)
        return file_data

class RequestHandler(BaseHTTPRequestHandler):
    file_dict = BitstreamServer()
    executor = ThreadPoolExecutor()

    async def write_and_flush(self, data):
        """
        Async function to write the data and flush
        """
        self.wfile.write(data)
        self.wfile.flush()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        # Extract parameters
        document_id = data['document_id']
        chunk_id = data['chunk_id']
        quality = data['quality']
        target_latency = data['target_latency']

        # Lookup in the dictionary
        
        file_data = self.file_dict.get(document_id, chunk_id, quality)

        if file_data is not None:
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.end_headers()

            # Send bytearray in chunks
            logging.info(f"Size is: {len(file_data)} bytes, should be sent within {target_latency} seconds")
            chunk_size_bytes = 256 * 1024 # 8MB chunk size
            start = time.time()
            data_len = len(file_data)
            for i in range(0, data_len, chunk_size_bytes):
                self.wfile.write(file_data[i:i+chunk_size_bytes])
                self.wfile.flush()
                curr_time = time.time()
                target_time = start + i * target_latency / data_len
                if i + chunk_size_bytes < len(file_data) and target_time > curr_time:
                    time.sleep(target_time - curr_time)  # Basic rate control

            end = time.time()
            logging.info(f"Real sending time is: {end-start} seconds")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Data not found for the given parameters")

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    #bs_server = BitstreamServer("/dataheart/yuhanl-share/unified.pkl")
    #run(handler_class = partial(RequestHandler, bs_server))
    run()

