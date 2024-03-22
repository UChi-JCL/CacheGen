"""
HuggingFace injections for KV cache

1. HuggingFace pipeline support kv_cache using kwargs
2. HuggingFace pipeline support processing GreedySearchDecoderOnlyOutput in _forward
3. multiple calls for the prefilling phase
"""
from typing import Dict
import torch
import gc
import numpy as np
from transformers.pipelines.text_generation import TextGenerationPipeline, ReturnType
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from transformers.pipelines.pt_utils import PipelineIterator
from transformers.utils.generic import ModelOutput

from .cachegen_interface import CacheGenController, merge_kv

def TextGenerationPipeline_forward(self: TextGenerationPipeline, 
                            model_inputs: Dict[str, torch.Tensor] , 
                            **generate_kwargs):
    """
    This function should be injected to transformers.pipelines.text_generation.TextGenerationPipeline
    Modifications:
        there will be a new item in the returning dictionary: "past_key_values" 
        if we are NOT using kv cache, this will be None
        otherwise, this will be a KVCache object (Tuple[Tuple[torch.Tensor, torch.Tensor]])
    """
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    if input_ids.shape[0] != 1:
        raise RuntimeError("Currently, only batch size 1 is supported for KV cache")

    # Allow empty prompts
    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]
    prompt_text = model_inputs.pop("prompt_text")

    # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
    # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
    prefix_length = generate_kwargs.pop("prefix_length", 0)
    if prefix_length > 0:
        has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
            "generation_config" in generate_kwargs
            and generate_kwargs["generation_config"].max_new_tokens is not None
        )
        if not has_max_new_tokens:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
            generate_kwargs["max_length"] += prefix_length
        has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
            "generation_config" in generate_kwargs
            and generate_kwargs["generation_config"].min_new_tokens is not None
        )
        if not has_min_new_tokens and "min_length" in generate_kwargs:
            generate_kwargs["min_length"] += prefix_length

    ''' Our modification: let's first do prefill here '''
    global cachegen_controller
    cachegen_configs = cachegen_controller.get(input_ids[0])
    prefill_kv = None
    for config in cachegen_configs:
        if config.is_kv:
            prefill_kv = merge_kv(prefill_kv, config.data, free_left = True, free_right = True)
        else:
            end_index = config.end_index
            generated = self.model.generate(inputs=input_ids[:, :end_index], 
                                            attention_mask=attention_mask[:, :end_index], 
                                            past_key_values=prefill_kv,
                                            return_dict_in_generate=True,
                                            max_length = 0,
                                            **generate_kwargs)
            del prefill_kv
            prefill_kv = generated.past_key_values 

    cachegen_controller.set(input_ids[0], prefill_kv)
    ''' End of our modification '''

    # BS x SL
    generated_sequence = self.model.generate(input_ids=input_ids, 
                                             attention_mask=attention_mask, 
                                             past_key_values = prefill_kv, 
                                             **generate_kwargs)
    '''
    Our modification:
        Original code: `generated_sequence` is just a torch.Tensor of sequences
        Now, since we want KV cache support, `generated_sequence` could be `GreedySearchDecoderOnlyOutput`
    '''
    if isinstance(generated_sequence, torch.Tensor):
        out_b = generated_sequence.shape[0]
        past_kv = None
    else:
        past_kv = generated_sequence.past_key_values
        out_b = generated_sequence.sequences.shape[0]
        generated_sequence = generated_sequence.sequences

    if self.framework == "pt":
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    elif self.framework == "tf":
        generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
    return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text, "past_key_values": past_kv}

def TextGenerationPipeline_ensure_tensor_on_device(self, inputs, device):
    """
    Override the Pipeline._ensure_tensor_on_device so that we don't copy KV cache to CPU
    """
    if isinstance(inputs, ModelOutput):
        return ModelOutput(
            {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        )
    elif isinstance(inputs, dict):
        return {name: self._ensure_tensor_on_device(tensor, device) if name != "past_key_values" else tensor 
                for name, tensor in inputs.items()}
    elif isinstance(inputs, list):
        return [self._ensure_tensor_on_device(item, device) for item in inputs]
    elif isinstance(inputs, tuple):
        return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
    elif isinstance(inputs, torch.Tensor):
        if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
            inputs = inputs.float()
        return inputs.to(device)
    else:
        return inputs

################################
######## Main interface ########
################################

_injection_dictionary = {}
cachegen_controller = None

def start_injection():
    """
    Prepare for all the injection code
    """
    global cachegen_controller
    cachegen_controller = CacheGenController.GetInstance()

    _injection_dictionary["TextGenerationPipeline._forward"] = TextGenerationPipeline._forward  
    TextGenerationPipeline._forward = TextGenerationPipeline_forward

    _injection_dictionary["TextGenerationPipeline._ensure_tensor_on_device"] = TextGenerationPipeline._ensure_tensor_on_device
    TextGenerationPipeline._ensure_tensor_on_device = TextGenerationPipeline_ensure_tensor_on_device


def clear_injection():
    """
    Clear all the injection code
    """
    if "TextGenerationPipeline._forward" in _injection_dictionary:
        TextGenerationPipeline._forward = _injection_dictionary["TextGenerationPipeline._forward"]

    if "TextGenerationPipeline._ensure_tensor_on_device" in _injection_dictionary:
        TextGenerationPipeline._ensure_tensor_on_device = _injection_dictionary["TextGenerationPipeline._ensure_tensor_on_device"]
