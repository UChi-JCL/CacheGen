import torch
from fastchat.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import argparse
import json
import numpy as np
import openai
from collections import Counter
import re 
import string 
def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


dataset2metric = {
    "nqa": qa_f1_score,
    "tqa": qa_f1_score,
}

MAX_API_RETRY = 5
REQ_TIME_GAP = 2
DATASET_TO_PATH = {
    "longchat": "test_data/longchat.jsonl",
    "tqa": "test_data/tqa.jsonl",
    "nqa": "test_data/nqa.jsonl"
}


def get_eval(user_prompt):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=500,
            )
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(e)
            time.sleep(5)
    print(f"Failed after {MAX_API_RETRY} retries.")
    return "error"
def scorer_e(dataset, predictions, answers, all_classes):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "tqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        scores += [score]
    
    return scores
def chatgpt_auto_eval(gt_result, cachegen_result):
    print("--------------- Start auto-evaluation, you should verify it does this correctly --------------")
    correct = 0
    user_prompt = f"I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple mistakes such as capitalization and punctuation. The ground truth is {gt_result}, the model prediction is {cachegen_result}"

    content = get_eval(user_prompt)

    _correct = content == "1"
    correct += _correct

    output_string = "correct" if _correct else "wrong"

    print(f"Label: {gt_result}, Predict: {cachegen_result} - auto-eval goes with {output_string}")

    # To avoid rate limit by OPENAI
    time.sleep(REQ_TIME_GAP)
    return correct

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)
def calculate_acc(dataset_name, prediction, label):
    if dataset_name == "longchat":
        return chatgpt_auto_eval(label[0], prediction)
    elif dataset_name == "nqa":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    elif dataset_name == "tqa":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    
    
def define_model_and_tokenizer(model_id, num_gpus=1, max_gpu_memory=48):
    """ Define the model and tokenizer
    """
    if model_id == "Yukang/LongAlpaca-70B-16k":
        from_pretrained_kwargs = {
                                'device_map': 'auto', 
                                'max_memory': {0: '45GiB', 
                                               1: '45GiB', 
                                               2: '45GiB', 
                                               3: '45GiB'}, 
                                'revision': 'main'}
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                load_in_8bit=True,
                **from_pretrained_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
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


def tensor_to_tuple(kv, layer_to_device_id):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}"), 
                       kv[i][1].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}")))
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
        kv[i][0] = quant_key
        kv[i][1] = quant_value
    kv = kv[:, :, :, :-10]
    return tensor_to_tuple(kv)

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases