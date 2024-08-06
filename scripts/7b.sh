export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
python main.py \
    --model_id  $MODEL \
    --save_dir ${SAVE_DIR}/${MODEL_ID}_$1_data/ \
    --start 0 \
    --end 50  \
    --num_gpus 1 \
    --dataset_name $1

export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
python run_cachegen.py \
    --model_id $MODEL \
    --save_dir ${SAVE_DIR}/${MODEL_ID}_$1_data \
    --start 0 \
    --end 50 \
    --num_gpus 1 \
    --encoded_dir ${SAVE_DIR}/encoded \
    --results_str cachegen \
    --results_dir ${MODEL_ID}_results/ \
    --dataset_name $1 \
    --calculate_metric $2
