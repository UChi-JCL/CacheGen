export MODEL=Yukang/LongAlpaca-70B-16k
export MODEL_ID=llama70b
python main.py \
    --model_id  $MODEL \
    --save_dir ${SAVE_DIR}/${MODEL_ID}_$1_data/ \
    --start 0 \
    --end 50  \
    --num_gpus 4 \
    --dataset_name $1

export MODEL=Yukang/LongAlpaca-70B-16k
export MODEL_ID=llama70b
python run_cachegen.py \
    --model_id $MODEL \
    --save_dir ${SAVE_DIR}/${MODEL_ID}_$1_data \
    --start 0 \
    --end 50 \
    --num_gpus 4 \
    --encoded_dir ${SAVE_DIR}/encoded \
    --results_str cachegen \
    --results_dir ${MODEL_ID}_results/ \
    --dataset_name $1 \
    --calculate_metric $2
