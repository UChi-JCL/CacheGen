export MODEL=Yukang/LongAlpaca-70B-16k
export MODEL_ID=llama70b
python main.py \
    --model_id  $MODEL \
    --save_dir ${SAVE_DIR}/${MODEL_ID}_$1_data/ \
    --start 0 \
    --end 50  \
    --num_gpus 4 \
    --dataset_name $1

python run_quantization_baseline.py \
    --model_id $MODEL \
    --dataset_name $1 \
    --num_gpus 4 \
    --save_dir  ${SAVE_DIR}/${MODEL_ID}_$1_data/ \
    --start 0 \
    --end 50 \
    --bins 256 \
    --calculate_metric $2

python run_vanilla.py \
    --model_id $MODEL \
    --dataset_name $1 \
    --num_gpus 4 \
    --save_dir  ${SAVE_DIR}/${MODEL_ID}_$1_data/ \
    --start 0 \
    --end 50 \
    --calculate_metric $2
