import requests
import logging
import time
import pickle
import hashlib
logging.basicConfig(format= '[%(asctime)s] %(levelname)s - %(message)s', level=logging.INFO)

def GbpsToKBytesPerSecond(gbps):
    return gbps * 125000

def send_request(document_id, chunk_id, quality, key_or_value, target_latency):
    """
    rate is KBytes per second
    Returns the received byte array, or None if there was an error
    Fields:
    - document_id: the document id
    - chunk_id: the chunk id
    - quality: the quality level of CacheGen 
    - target_latency: the target latency
    - key_or_value: 0 for key and 1 for value
    
    """
    # TODO: change "localhost" to the IP address of the storage server (CPU server)
    url = 'http://localhost:8000'

    fixed_overhead = 0.02 # 20 ms
    data = {
        'document_id': document_id,
        'chunk_id': chunk_id,
        'quality': quality,
        'key_or_value': key_or_value,
        'target_latency': target_latency - fixed_overhead
    }

    t1 = time.time()
    response = requests.post(url, json=data)
    t2 = time.time()
    logging.info(f"Time to get the response: {t2 - t1}")
    logging.info(f"Rate: {len(response.content) / (t2 - t1) / 1e6 * 8} Mbps")

    if response.status_code == 200:
        logging.info(f"Successfully received {len(response.content)} bytes")
        return pickle.loads(response.content)
    else:
        logging.error("Error: ", response.text)
        return None

if __name__ == '__main__':
    #rate = GbpsToKBytesPerSecond(0.6)
    target_latency = 0.2
    start = time.time()
    send_request(0, 0, 0, 0, target_latency)  # Example usage
    end = time.time()

