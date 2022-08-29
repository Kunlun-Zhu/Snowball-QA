export MAX_SEQ_LEN=512
export EPOCHS=2
export USE_CUDA=True
export BATCH_SIZE=1024
export LR=0.00001
export WEIGHT_DECAY=0.0001
export SAVE_INTERVAL=20
export PRINT_INTERVAL=40000
export SAVE_PATH=./output/
export TRAIN_DATA_DIR=./data/dureader-robust/demo_train.jsonl
export TEST_DATA_DIR=./data/answer_extractor_data/demo.jsonl
export PRETRAINED_PATH=./data/pretrained_model/bert-base-multilingual-cased/
export GPU_IDS=0
export INFERENCE_PATH=/yinxr/hx/cpm-live/Answer_extractor/output_combine/8_combine.jsonl

python main.py \
    --max_seq_len ${MAX_SEQ_LEN} \
    --epochs ${EPOCHS} \
    --use_cuda ${USE_CUDA} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --save_interval ${SAVE_INTERVAL} \
    --print_interval ${PRINT_INTERVAL} \
    --save_path ${SAVE_PATH} \
    --train_data_dir ${TRAIN_DATA_DIR} \
    --test_data_dir ${TEST_DATA_DIR} \
    --pretrained_path ${PRETRAINED_PATH} \
    --gpu_ids ${GPU_IDS} \
    --do_train False \
    --do_predict True \
    --eval_model_path /yinxr/hx/cpm-live/Answer_extractor/model/mixed_train_epoch1.pt \
    --inference_path ${INFERENCE_PATH}
