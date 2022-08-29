import sys
import jsonlines
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForQuestionAnswering
import torch
import json
from tqdm import tqdm
import math
import torch
import logging
import numpy as np
from dataset import PertIterableDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import argparse
from utils import boolean_string


parser = argparse.ArgumentParser()

parser.add_argument("--max_seq_len", default=512, type=int, required=True, help="sequence length.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=16, type=int, required=True, help="batch size.")
parser.add_argument("--input_data_dir", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str,
                    required=True, help="save directory.")
parser.add_argument("--output_data_dir", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str,
                    required=True, help="save directory.")
parser.add_argument("--pretrained_path",
                    default='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base-multilingual-cased/', type=str,
                    required=True, help="pretrained path.")
parser.add_argument("--gpu_ids", default='6', type=str, required=True, help="gpu ids.")

args = parser.parse_args()


def predict_answer(model, dataset, tokenizer):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with jsonlines.open(args.output_data_dir, mode='w') as writer:
        writer._flush = True
        for batch_data in tqdm(dataloader):
            batch_size = batch_data["input_ids"].size(0)
            input_idss = batch_data["input_ids"]
            token_type_ids = batch_data["token_type_ids"]
            attention_mask = batch_data["attention_mask"]
            outputs = model(input_ids=input_idss, token_type_ids=token_type_ids, attention_mask=attention_mask)
            answer_start_scoress = outputs["start_logits"]
            answer_end_scoress = outputs["end_logits"]
            for i in range(batch_size):
                answer_start_scores = answer_start_scoress[i]
                answer_end_scores = answer_end_scoress[i]
                passage = batch_data["passage"][i]
                question = batch_data["question"][i]
                ori_answer = batch_data["answer"][i]
                ori_answer_start = passage.find(ori_answer)
                ori_answer_end = ori_answer_start + len(ori_answer)
                answer_start = torch.argmax(
                    answer_start_scores
                )  # 获得最可能是答案开始的token的下标
                answer_end = torch.argmax(answer_end_scores) + 1  # 获得最可能是答案结束的token的下标

                input_ids = input_idss[i]
                # print(text_tokens)
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])).replace(" ", "")
                answer_start_clean = passage.find(answer)
                answer_end_clean = answer_start_clean + len(answer)
                if answer == '' or '[CLS]' in answer:
                    continue
                if answer in ori_answer or ori_answer in answer:
                    final_answer = answer
                    start = answer_start_clean
                elif ori_answer_start < answer_end_clean < ori_answer_end:
                    final_answer = passage[answer_start_clean: ori_answer_end]
                    start = answer_start_clean
                elif answer_start_clean < ori_answer_end < answer_end_clean:
                    final_answer = passage[ori_answer_start: answer_end_clean]
                    start = ori_answer_start
                else:
                    continue
                writer.write({"passage": passage, "extracted_answer": final_answer, "generated_question": question,
                              "answer_start": start})


if __name__ == '__main__':
    gid = int(args.gpu_ids[0])
    device_ids = list()
    for gpu_id in args.gpu_ids:
        device_ids.append(int(gpu_id))
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_path, return_dict=True)

    device = torch.device('cuda:{}'.format(gid))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
    dataset = PertIterableDataset(args.input_data_dir, args.max_seq_len, args.pretrained_path)

    predict_answer(model, dataset, tokenizer)
