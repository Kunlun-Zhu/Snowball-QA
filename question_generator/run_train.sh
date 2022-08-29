export max_seq_len=512
export batch_size=20
export weight_decay=0.0001
export epochs=10
export logging_steps=50
export evaluation_strategy="steps"
export eval_steps=500
export load_best_model_at_end=True
export learning_rate=1e-5
export output_dir="test_squad"
export save_total_limit=5
export lr_scheduler_type='linear'
export warmup_ratio=0.1
export warmup_steps=10
export gradient_accumulation_steps=1
export dataloader_num_workers=4
export pretrained_path=/liuzyai04/BMKG/zhukunlun/QA-gen/pretrained_model/mengzi-t5/
export train_data_dir=/liuzyai04/BMKG/zhukunlun/QA-gen/data/combine_all_zh.jsonl
export dev_data_dir=/liuzyai04/BMKG/zhukunlun/QA-gen/data/dev_qg.jsonl

python3 -u train.py \
--max_seq_len ${max_seq_len} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--logging_steps ${logging_steps} \
--evaluation_strategy ${evaluation_strategy} \
--eval_steps ${eval_steps} \
--learning_rate ${learning_rate} \
--output_dir ${output_dir} \
--save_total_limit ${save_total_limit} \
--lr_scheduler_type ${lr_scheduler_type} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--dataloader_num_workers ${dataloader_num_workers} \
--weight_decay ${weight_decay} \
--train_data_dir ${train_data_dir} \
--dev_data_dir ${dev_data_dir} \
--pretrained_path ${pretrained_path} \
--warmup_ratio ${warmup_ratio} \
--warmup_steps ${warmup_steps}
