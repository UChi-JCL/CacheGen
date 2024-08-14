In this section, we provide the scripts for running the experiments in Figure 8/9 for a range of models and datasets. 
Note that for the artifact evaluation, we only provided **a small subset** of data points for easier testing. 

### Mistral-7B
Before running the encoder of CacheGen, first runs the following code to generate the KV caches. The following code generates the KV caches for the **LongChat dataset** (in Fig. 8/9) on Mistral-7B model.
Note that this assumes you have **one A40 GPU with 48GB GPU memeory**. 


```
export SAVE_DIR=.
bash scripts/7b.sh longchat 0
```

This will output CacheGen's size of KV cache on the LongChat dataset. 

If you want to test the accuracy, you need to set the **OPENAI api key** (You can obtain the key here: https://platform.openai.com/api-keys)
```
export OPENAI_API_KEY=<YOUR API KEY>
export SAVE_DIR=.
bash scripts/7b.sh longchat 1
```
This will output the accuracy and size of KV cache on the LongChat dataset. 

### Llama-70B

**This requires you to have 4 A40 GPUs!**

First to run with the **longchat** dataset,  run the following code:
```
export SAVE_DIR=.
bash scripts/70b.sh longchat 0
```

If you want to test the accuracy, you need to set the **OPENAI api key**
```
export OPENAI_API_KEY=<YOUR API KEY>
export SAVE_DIR=.
bash scripts/70b.sh longchat 1
```
This will output the accuracy and size of KV cache on the LongChat dataset. 


To run with the **NarrativeQA** dataset, run the following code:
```
export SAVE_DIR=.
bash scripts/70b.sh nqa 1
```
This will output the accuracy and size of KV cache on the NarrativeQA dataset. 

To run with the **TriviaQA** dataset, run the following code:
```
export SAVE_DIR=.
bash scripts/70b.sh tqa 1
```
This will output the accuracy and size of KV cache on the TriviaQA dataset. 
