export gpu=0
export max_seq_len=512
export batch_size=100
export beam_nums=4
export output_dir=/data/private/wanghuadong/liangshihao/MengZi/results/qg_demo.jsonl
export dev_data_dir=/data/private/wanghuadong/liangshihao/QA/data/paq/dev_qg.jsonl
export pretrained_model_path=/data/private/wanghuadong/liangshihao/MengZi/pretrained_model/mengzi-t5/
export ckpt_path=iter1/best/

python3 -u predict.py \
--gpu ${gpu} \
--max_seq_len ${max_seq_len} \
--batch_size ${batch_size} \
--beam_nums ${beam_nums} \
--output_dir ${output_dir} \
--dev_data_dir ${dev_data_dir} \
--pretrained_model_path ${pretrained_model_path} \
--ckpt_path ${ckpt_path}