# export MODEL=Yukang/LongAlpaca-70B-16k
export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
python run_quantization_baseline.py \
--model_id ${MODEL} \
--save_dir /data/${MODEL_ID}_data \
--start 0 \
--end 60 \
--bins 15 \
--num_gpus 1 \
--results_str quant_4bit_agg \
--max_gpu_memory 70 \
--results_dir ${MODEL_ID}_results/ --path_to_context test_data/16_prompts

# python run_cachegen.py \
# --model_id mistralai/Mistral-7B-Instruct-v0.1 \
# --save_dir /data/instruct_v0.1_data \
# --start 0 \
# --end 50 \
# --encoded_dir encoded \
# --results_str cachegen \
# --results_dir results/ --path_to_context 20_prompts_instruct
