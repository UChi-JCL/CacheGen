import torch
from src.utils import *
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
class CacheGenEngine:
    def __init__(self, model_id):
        self.model_id = model_id
        self.nchunks = 0
    def chunk_kv(self, kv, doc_id, chunk_size=1024, encoded_dir=None):
        """ The function to implement the chunking logic of CacheGen. 
        """
        ntokens = kv.shape[-2]
        nchunks = ntokens // chunk_size
        self.nchunks = nchunks
        for i in range(nchunks):
            chunk = kv[:, :, :, i*chunk_size:(i+1)*chunk_size]
            lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=chunk.shape[-2])
            meta_data = LMCacheEngineMetadata(model_name=self.model_id, fmt="huggingface", world_size=1, worker_id=0)
            cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
            bytes = cachegen_serializer.to_bytes(chunk)
            torch.save(torch.frombuffer(bytes, dtype=torch.uint8), f"{encoded_dir}/doc_{doc_id}_chunk_{i}.pt" )
    def decode_kv(self, encoded_dir, doc_id, chunk_size):
        """ The function to implement the decoding logic of CacheGen. 
        """
        decoded_kv = []
        for i in range(self.nchunks ):
            bytes = torch.load(f"{encoded_dir}/doc_{doc_id}_chunk_{i}.pt")
            lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
            meta_data = LMCacheEngineMetadata(model_name=self.model_id, fmt="huggingface", world_size=1, worker_id=0)
            deserializer = CacheGenDeserializer(lmcache_config, meta_data)
            chunk = deserializer.from_bytes(bytes.numpy().tobytes())
            decoded_kv.append(chunk)
        
        decoded_kv = torch.cat(decoded_kv, dim=-2)
        return decoded_kv