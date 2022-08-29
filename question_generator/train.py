import json
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from tqdm import tqdm
import torch
import random
import argparse
from dataset import Seq2SeqDataset, DataCollatorForSeq2Seq

import os
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()

parser.add_argument("--lr_scheduler_type", default='constant', type=str, required=True, help="gpu or cpu.")
parser.add_argument("--max_seq_len", default=512, type=int, required=True, help="sequence length.")
parser.add_argument("--epochs", default=5, type=int, required=True, help="epoch.")
parser.add_argument("--batch_size", default=16, type=int, required=True, help="batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=float, required=True, help="lr.")
parser.add_argument("--weight_decay", default=0.0001, type=float, required=True, help="lr.")
parser.add_argument("--save_total_limit", default=5, type=int, required=True, help="ckpt nums")
parser.add_argument("--output_dir", default='/liuzyai04/BMKG/zhukunlun/QA-gen/output/', type=str, required=True, help="save directory.")
parser.add_argument("--train_data_dir", default='/liuzyai04/BMKG/zhukunlun/QA-gen/data/', type=str, required=True, help="save directory.")
parser.add_argument("--dev_data_dir", default='/liuzyai04/BMKG/zhukunlun/QA-gen/data/', type=str, required=True, help="save directory.")
parser.add_argument("--pretrained_path", default='Langboat/mengzi-t5-base', type=str, required=True, help="pretrained path.")
parser.add_argument("--lr_scheduler_type", default='constant', type=str, required=True, help="gpu or cpu.")
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, required=True, help="gpu or cpu.")
parser.add_argument("--dataloader_num_workers", default=0, type=int, required=True, help="gpu or cpu.")
parser.add_argument("--evaluation_strategy", default='steps', type=str, required=True, help="gpu or cpu.")
parser.add_argument("--eval_steps", default=100, type=int, required=True, help="gpu or cpu.")
parser.add_argument("--logging_steps", default=10, type=int, required=True, help="gpu or cpu.")
parser.add_argument("--warmup_ratio", default=0.1, type=float, required=True, help="gpu or cpu.")
parser.add_argument("--warmup_steps", default=100, type=int, required=True, help="gpu or cpu.")
args = parser.parse_args()




if __name__ == '__main__':

    # model_path = "Langboat/mengzi-t5-base"

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path).cuda()
    tokenizer.add_tokens(['<hl>'])
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model.cuda()

    trainset = Seq2SeqDataset(args.train_data_dir)
    devset = Seq2SeqDataset(args.dev_data_dir)
    collator = DataCollatorForSeq2Seq(tokenizer, max_length=args.max_seq_len)

    output_dir = "iter2" # 模型checkpoint的保存目录
    training_args = TrainingArguments(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_steps=args.logging_steps,
            weight_decay=args.weight_decay,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_num_workers=args.dataloader_num_workers)
    print('Training Arguments ...')
    print(training_args)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=trainset,
        eval_dataset=devset
    )

    trainer.train()
    trainer.save_model("iter2/best") # 保存最好的模型