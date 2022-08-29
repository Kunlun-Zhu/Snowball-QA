""" data reader for Answer generation
"""
import os
import csv
import sys
import json
import tqdm
import math
import torch
import logging
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer
# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#     datefmt='%m/%d/%Y %H:%M:%S')
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.info(logger.getEffectiveLevel())

torch.manual_seed(1)


class BaseIterableDataset(IterableDataset):
    def __init__(self, file_path, max_seq_len, pretrained_path):
        self.file_path = file_path
        self.info = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end = self.info['end']

        if "roberta" in pretrained_path:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
            self.cls_id = self.tokenizer.convert_tokens_to_ids("<s>")
            self.sep_id = self.tokenizer.convert_tokens_to_ids("</s>")
            self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
            self.mask_id = self.tokenizer.convert_tokens_to_ids("<mask>")
            self.unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
            self.cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
            self.sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
            self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.max_seq_len = max_seq_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            iter_start = self.start
            iter_end = self.end
        else:  # multiple workers
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        sample_iterator = self._sample_generator(iter_start, iter_end)
        return sample_iterator

    def _sample_generator(self, start, end):
        def get_answer_position(src, tgt):
            tgt_pos = 0
            for pos, token in enumerate(src):
                if tgt_pos >= len(tgt):
                    return pos - len(tgt) + 1
                if token == tgt[tgt_pos]:
                    if tgt_pos == len(tgt) - 1:
                        return pos - len(tgt) + 1
                    else:
                        tgt_pos += 1
                elif token == tgt[0]:
                    tgt_pos = 1
                else:
                    tgt_pos = 0
                    continue
            return -1

        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i < start: continue
                if i >= end: return StopIteration()

                json_data = json.loads(line)
                passage_text = json_data["passage"]
                passage = self.tokenizer(passage_text,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=self.max_seq_len,
                                         return_tensors="pt",
                                         )
                pas_ids = self.tokenizer.encode(passage_text)
                label_ids = [0] * min(self.max_seq_len, len(pas_ids))
                for index in range(len(json_data["answer"])):
                    answer_text = json_data["answer"][index]
                    ans_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer_text))

                    ans_pos = get_answer_position(pas_ids, ans_ids)
                    # print(ans_pos)
                    if ans_pos == -1 or ans_pos + len(ans_ids) > self.max_seq_len:
                        continue
                    for i in range(len(ans_ids)):
                        label_ids[ans_pos + i] = 1
                        assert pas_ids[ans_pos + i] == ans_ids[i]
                label_ids.extend([-1] * (self.max_seq_len - len(label_ids)))
                assert len(label_ids) == self.max_seq_len
                sample = {"input_ids": passage["input_ids"].squeeze(), "label_ids": torch.tensor(label_ids),
                          "input_mask": passage["attention_mask"].squeeze()}
                yield sample

    def _get_file_info(self,
                       file_path
                       ):
        info = {
            "start": 1,
            "end": 0
        }
        with open(file_path, 'r') as fin:
            for _ in enumerate(fin):
                info['end'] += 1
        return info

    def get_tokenizer(self):
        return self.tokenizer

    def __len__(self):
        return self.end - self.start

class BertIterableDataset(BaseIterableDataset):

    def __init__(self, file_path, max_seq_len, pretrained_path):
        super(BertIterableDataset, self).__init__(file_path, max_seq_len, pretrained_path)

    def _sample_generator(self, start, end):
        def get_answer_position(src, tgt):
            tgt_pos = 0
            for pos, token in enumerate(src):
                if tgt_pos >= len(tgt):
                    return pos - len(tgt) + 1
                if token == tgt[tgt_pos]:
                    if tgt_pos == len(tgt) - 1:
                        return pos - len(tgt) + 1
                    else:
                        tgt_pos += 1
                elif token == tgt[0]:
                    tgt_pos = 1
                else:
                    tgt_pos = 0
                    continue
            return -1
        with open(self.file_path, 'r') as fin:                                  
            for i, line in enumerate(fin):                                      
                if i < start: continue                                          
                if i >= end: return StopIteration()  

                json_data = json.loads(line)
                passage_text = json_data["passage"]
                passage = self.tokenizer(passage_text,
                                      truncation=True,
                                      padding='max_length',
                                      max_length=self.max_seq_len,
                                      return_tensors="pt",
                                      )
                pas_ids = self.tokenizer.encode(passage_text)
                label_ids = [0] * min(self.max_seq_len, len(pas_ids))
                for index in range(len(json_data["answer"])):
                    answer_text = json_data["answer"][index]
                    ans_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer_text))
                    
                    ans_pos = get_answer_position(pas_ids, ans_ids)
                    # print(ans_pos)
                    if ans_pos == -1 or ans_pos + len(ans_ids) > self.max_seq_len:
                        continue
                    for i in range(len(ans_ids)):
                        label_ids[ans_pos + i] = 1
                        assert pas_ids[ans_pos + i] == ans_ids[i]
                label_ids.extend([-1] * (self.max_seq_len - len(label_ids)))
                assert len(label_ids) == self.max_seq_len
                sample = {"input_ids": passage["input_ids"].squeeze(), "label_ids": torch.tensor(label_ids), "input_mask":passage["attention_mask"].squeeze()}
                yield sample
    

class PertIterableDataset(BaseIterableDataset):

    def __init__(self, file_path, max_seq_len, pretrained_path):
        super(PertIterableDataset, self).__init__(file_path, max_seq_len, pretrained_path)

    def _sample_generator(self, start, end):
        def get_answer_position(src, tgt):
            tgt_pos = 0
            for pos, token in enumerate(src):
                if tgt_pos >= len(tgt):
                    return pos - len(tgt) + 1
                if token == tgt[tgt_pos]:
                    if tgt_pos == len(tgt) - 1:
                        return pos - len(tgt) + 1
                    else:
                        tgt_pos += 1
                elif token == tgt[0]:
                    tgt_pos = 1
                else:
                    tgt_pos = 0
                    continue
            return -1
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i < start: continue
                if i >= end: return StopIteration()

                json_data = json.loads(line)
                passage_text = json_data["passage"]
                answer_text = json_data["extracted_answer"]
                question_text = json_data["generated_question"]
                answer_start = passage_text.find('<hl>')
                passage_text = passage_text.replace('<hl>', '')
                inputs = self.tokenizer(question_text, passage_text,
                                      add_special_tokens=True,
                                      truncation="only_second",
                                      padding='max_length',
                                      max_length=512,
                                      return_tensors="pt")

                sample = {"input_ids": inputs["input_ids"].squeeze(), "token_type_ids": inputs["token_type_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(),
                          "passage":passage_text, "question":question_text, "answer":answer_text, "answer_start":answer_start}
                yield sample
