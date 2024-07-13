# export MODEL=mistral-community/Mistral-7B-v0.2
# export MODEL_ID=mistral7b
#  python main.py \
#  --model_id  $MODEL \
#  --save_dir /data/${MODEL_ID}_data/ \
#  --path_to_context test_data/16_prompts/ \
#  --start 0 \
#  --end 60  \
#  --num_gpus 1 \
#  --max_gpu_memory 70

export MODEL=Yukang/LongAlpaca-70B-16k
export MODEL_ID=mistral7b
 python main.py \
 --model_id  $MODEL \
 --save_dir /data/${MODEL_ID}_data/ \
 --path_to_context test_data/16_prompts/ \
 --start 0 \
 --end 60  \
 --num_gpus 2 \
 --max_gpu_memory 70