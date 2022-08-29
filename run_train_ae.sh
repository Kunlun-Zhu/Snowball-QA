export MAX_SEQ_LEN=512
export EPOCHS=5
export USE_CUDA=True
export BATCH_SIZE=16
export LR=0.00001
export WEIGHT_DECAY=0.0001
export SAVE_INTERVAL=10
export PRINT_INTERVAL=2000
export SAVE_PATH=./output/
export TRAIN_DATA_DIR=/yinxr/hx/cpm-live/Answer_extractor/data/combine_all_multi_answer.jsonl
export TEST_DATA_DIR=/yinxr/hx/cpm-live/Answer_extractor/data/test.jsonl
export PRETRAINED_PATH=hfl/chinese-bert-wwm-ext
export GPU_IDS=7
export INFERENCE_PATH=/data/private/wanghuadong/liangshihao/QA/output/answers_demo.json


# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

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
    --do_train True \
    --do_predict False \
    --eval_model_path /data/private/wanghuadong/liangshihao/QA/output/mixed_train_epoch1.pt \
    --inference_path ${INFERENCE_PATH}
