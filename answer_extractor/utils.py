from lib2to3.pgen2 import token
import jieba
import torch
import torch.nn as nn
import jieba.posseg as psg
import json
from tqdm import tqdm


def read_json(input_file: str) -> list:
    '''
    读取json文件每行是一个json字段

    Args:
        input_file:文件名

    Returns:
        lines
    '''
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return list(map(json.loads, tqdm(lines, desc='Reading...')))


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


# add_list = ['[CLS]', '[SEP]', '[UNK]']
# for w in add_list:
#     jieba.add_word(w, freq=200000)

def get_start_end(tokens):
    start_list = [0]
    end_list = []
    index = 0
    for item in tokens:
        end_list.append(start_list[index] + len(item) - 1)
        start_list.append(end_list[index] + 1)
        index += 1
    start_list.pop()
    return start_list, end_list


def correct_start(start_list: list, pos: int) -> int:
    if pos in start_list:
        return pos
    for index, l_pos in enumerate(start_list):
        if l_pos > pos:
            return start_list[index - 1]
    raise ValueError('Out of range Error')


def correct_end(end_list: list, pos: int) -> int:
    # print(pos)
    # print(end_li0st)
    if pos in end_list:
        return pos
    for index, l_pos in enumerate(end_list):
        if l_pos > pos:
            return end_list[index]
    return end_list[-1]
    raise ValueError('Out of range Error')


def correct_start_tag(start_list: list, pos: int, tag_list: list) -> int:
    # combine the same tag with the first word into the answer
    index_s = start_list.index(pos) - 1
    while index_s >= 0:
        if (tag_list[index_s] == tag_list[index_s + 1]):
            pos = start_list[index_s]
        else:
            break
        index_s -= 1
    return pos
    raise ValueError('Out of range Error')


def correct_end_tag(end_list: list, pos: int, tag_list: list) -> int:
    # combine the same tag with the last word into the answer
    index_e = end_list.index(pos) + 1
    tot_len = len(tag_list)
    while index_e < tot_len:
        if (tag_list[index_e] == tag_list[index_e - 1]):
            pos = end_list[index_e]
        else:
            break
        index_e += 1
    not_end_list = ['c', 'dg', 'd', 'h', 'p', 'u', 'v', 'vd']  # list of tag should not be the end
    re_tagging = (index_e < tot_len and (str(tag_list[index_e - 1]) in not_end_list))  # the end tag need to re-process
    while index_e < tot_len and (str(tag_list[index_e - 1]) in not_end_list):
        pos = end_list[index_e]
        index_e += 1
    if re_tagging:
        while index_e < tot_len:
            if (tag_list[index_e] == tag_list[index_e - 1]):
                pos = end_list[index_e]
            else:
                break
            index_e += 1
    return pos


def get_pos(answer: str, passage: str):
    start = passage.find(answer)
    end = start + len(answer) - 1
    return start, end


def clean_text(text: str) -> str:
    return text.replace('[PAD]', '').replace(' ', '').replace('[CLS]', '').replace('[SEP]', '').replace('[UNK]', '')


filter_set1 = list("`~!@#$^&*()_+-=,./;[]\<>?{/}|，。、；‘、的了")
filter_set2 = list("`~!@#$^&*()_+-=,./;[]\<>?{/}|，。、‘、了")


def clean_answer(answer: str) -> str:
    try:
        while answer[0] in filter_set1:
            answer = answer[1:]
        while answer[-1] in filter_set2:
            answer = answer[:-1]
    except:
        return ''
    return answer


def extract_answer(tokenizer, src, logit, label_id, min_span_len, logits) -> dict:
    tokens, tag_list = [], []
    logit = logit[1:]
    logits = logits[1:, 1]
    passage = clean_text(tokenizer.decode(src))
    passage_len = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(passage)))
    seg_list = psg.cut(passage)
    for w, t in seg_list:
        tokens.append(w)
        tag_list.append(t)
    # tokens = jieba.lcut(passage)
    start_list, end_list = get_start_end(tokens)
    ans_set = set()
    output_dict = {}
    span_cnt = 0
    ans_list = []
    sort_ans_dict = {}
    for idx, logi in enumerate(logit):
        if idx > passage_len:
            break
        if logi == 1:
            span_cnt += 1
        elif span_cnt >= min_span_len:
            ans_dict = {}
            ans = src[idx - span_cnt: idx]
            ans = tokenizer.decode(ans)
            ans = ans.replace(' ', '')
            # 答案后处理
            # start_pos = idx - span_cnt
            # end_pos = idx
            start_pos, end_pos = get_pos(ans, passage)
            span_cnt = 0
            try:
                start_pos = correct_start(start_list, start_pos)
                start_pos = correct_start_tag(start_list, start_pos, tag_list)
            except:
                print(start_pos)
                print(start_list)
                continue
            if start_pos == 0:
                continue
            end_pos = correct_end(end_list, end_pos)
            end_pos = correct_end_tag(end_list, end_pos, tag_list)
            ans = passage[start_pos: end_pos + 1]
            ans = clean_answer(ans)
            if ans == '' or (end_pos - start_pos <= 1):
                continue
            if ans not in ans_set:
                # print(logits[start_pos: end_pos].shape)
                # print(sum(logits[start_pos: end_pos]) / (end_pos - start_pos))
                ans_score = sum(logits[start_pos: end_pos]) / (end_pos - start_pos)
                sort_ans_dict[ans + '\t' + str(start_pos)] = ans_score
                # ans_list.append(ans_dict)
                ans_set.add(ans)
    sort_ans_dict = dict(sorted(sort_ans_dict.items(), key=lambda x: x[1], reverse=True))
    cnt = 0
    for key, value in sort_ans_dict.items():
        if cnt >= 5:
            break
        ans, start_pos = key.split('\t')
        ans_dict = {'answer_text': ans, 'answer_score': value, 'answer_start': start_pos}
        ans_list.append(ans_dict)
        cnt += 1

    label = tokenizer.decode(src[(label_id == 1)])
    label = label.replace(' ', '')
    src = clean_text(tokenizer.decode(src))
    output_dict["passage"] = src
    output_dict["origin_answer"] = label
    output_dict["candidate_answers"] = ans_list
    return output_dict
