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

def preprocess(item: dict):
    question = item["question"]
    answer = item["answer"]
    passage = item["passage"]
    language = item["language"]
    offset = int(item["answer_start"])
    passage = list(passage)
    passage.insert(offset, "<hl>")       
    passage.insert(offset + len(answer) + 1, "<hl>")
    passage = ''.join(passage)
    passage.replace("[UNK]", '')
    passage.replace("[SEP]", '')
    # print(passage)
    # print(answer)

    # if language == "zh":
    #     passage = "给定文档与答案，生成问题。：" + passage
    #     passage += ("。答案：" + answer)
    # else:
    #     passage = "Given Context and Answer, generate question. Context: " + passage
    #     passage += (". Answer: " + answer)
    return passage, question, answer
