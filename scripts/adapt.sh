export MODEL=mistral-community/Mistral-7B-v0.2

python run_adaptation.py \
--model_id ${MODEL} \
--num_gpus 1 \
--dataset_name longchat \
--save_dir ${SAVE_DIR}/${MODEL}_encoded \
--start 0 \
--end 50 \
--slo 0.5 \
--encode


python run_adaptation.py \
--model_id ${MODEL} \
--num_gpus 1 \
--dataset_name longchat \
--save_dir ${SAVE_DIR}/${MODEL}_encoded \
--start 0 \
--end 50 \
--slo 0.5 \
--chunk_size 1500 \
--total_traces 5 \
--calculate_metric 1 
