from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
from langchain.prompts import PromptTemplate
import huggingface_injections as hf_injection



model_id = "lmsys/longchat-7b-16k"
hf = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task = "text-generation",
        device = 0,
        model_kwargs = {
            "max_length": 0,
            "load_in_8bit": True
        },
    )

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

#chain = prompt | hf
chain = hf

with open("../7k_prompts/0.txt", "r") as fin:
    question = fin.read() 
question = question[:len(question) // 2] # make it fit the GPU memory

def run_chain(show = False):
    st = time.time()
    response = chain.invoke(question)
    ed = time.time()
    if show:
        print("The response is: ", response)
    return ed - st

def profile_func():
    total_times = [run_chain(show = True) for i in range(2)]
    print("First call", total_times[0])
    print("Second call", total_times[1])

#import cProfile
#cProfile.run("profile_func()", "latest.prof")

''' warm up before injection '''
run_chain(show=True)

hf_injection.start_injection()
profile_func()
hf_injection.clear_injection()
