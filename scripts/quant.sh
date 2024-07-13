# 
export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
python run_quantization_baseline.py \
--model_id ${MODEL} \
--save_dir /data/${MODEL_ID}_data \
--start 0 \
--end 60 \
--bins 256 \
--num_gpus 1 \
--results_str quant_8bit \
--max_gpu_memory 70 \
--results_dir ${MODEL_ID}_results/ --path_to_context test_data/16_prompts
