
python run_cachegen.py \
--model_id mistral-community/Mistral-7B-v0.2 \
--save_dir /data/mistral7b_data \
--start 0 \
--end 50 \
--encoded_dir encoded \
--results_str cachegen \
--results_dir results/ --path_to_context test_data/16_prompts
