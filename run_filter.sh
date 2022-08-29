export MAX_SEQ_LEN=512
export USE_CUDA=True
export BATCH_SIZE=40
export INPUT_DATA_DIR=/yinxr/hx/cpm-live/QA-Gen/results/5_combine_generate_filted.jsonl
export OUTPUT_DATA_DIR=/yinxr/hx/cpm-live/QA-Gen/results/5_final.jsonl
export PRETRAINED_PATH=/yinxr/hx/cpm-live/QA-Gen/pretrained_model/pert/
export GPU_IDS=2

# export CUDA_VISIBLE_DEVICES=1,3

python pert_mrc.py \
    --max_seq_len ${MAX_SEQ_LEN} \
    --use_cuda ${USE_CUDA} \
    --batch_size ${BATCH_SIZE} \
    --input_data_dir ${INPUT_DATA_DIR} \
    --output_data_dir ${OUTPUT_DATA_DIR} \
    --pretrained_path ${PRETRAINED_PATH} \
    --gpu_ids ${GPU_IDS}
