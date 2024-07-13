export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
python run_cachegen.py \
--model_id $MODEL \
--save_dir /data/${MODEL_ID}_data \
--start 0 \
--end 50 \
--encoded_dir encoded \
--results_str cachegen \
--results_dir ${MODEL_ID}_results/ --path_to_context test_data/16_prompts

