import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch
import argparse
import jsonlines
from dataset import preprocess, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, required=True, help="sequence length.")
parser.add_argument("--max_seq_len", default=512, type=int, required=True, help="sequence length.")
parser.add_argument("--batch_size", default=8, type=int, required=True, help="sequence length.")
parser.add_argument("--beam_nums", default=8, type=int, required=True, help="sequence length.")
parser.add_argument("--output_dir", default='/data/private/wanghuadong/liangshihao/QA/output/', type=str, required=True, help="save directory.")
parser.add_argument("--dev_data_dir", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str, required=True, help="save directory.")
parser.add_argument("--pretrained_model_path", default='Langboat/mengzi-t5-base', type=str, required=True, help="pretrained path.")
parser.add_argument("--ckpt_path", default='Langboat/mengzi-t5-base', type=str, required=True, help="pretrained path.")
args = parser.parse_args()


torch.cuda.set_device(int(args.gpu))

if __name__ == '__main__':
    test_data = read_json(args.dev_data_dir)
    test_data = map(preprocess, test_data)
    writer = jsonlines.open(args.output_dir, mode='w')
    writer._flush = True
    batch_size = args.batch_size
    inputs, refs, answers = zip(*test_data)

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path).cuda()
    tokenizer.add_tokens(['<hl>'])
    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.ckpt_path + "/pytorch_model.bin", map_location=torch.device('cuda:{}'.format(args.gpu)))
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    state_dict=None
    new_state_dict=None

    model.eval()
    kwargs = {"num_beams": args.beam_nums}
    input_len = len(inputs)
    last_passage = ''
    q_list = []
    for start in tqdm(range(0, input_len, batch_size)):
        batch = inputs[start:start + batch_size]
        output_ans = []
        input_tensor = tokenizer(batch, return_tensors="pt", truncation=True, padding=True,
                                 max_length=512).input_ids.cuda()
        batch_generate = model.generate(input_ids=input_tensor, **kwargs)
        output_ans.extend(batch_generate)
        batch_output = tokenizer.batch_decode(output_ans, skip_special_tokens=True)
        for i in range(batch_size):
            if start + i > input_len - 1:
                break
            output_dict = {}
            passage = inputs[start + i].replace("<hl>", "")
            question = batch_output[i]
            answer = answers[start + i]
            if last_passage == passage:
                if batch_output[i] not in q_list:
                    q_list.append(question)
                    writer.write({"passage": passage, "generated_question": question, "extracted_answer": answer})
            else:
                q_list = [question]
                writer.write({"passage": passage, "generated_question": question, "extracted_answer": answer})

            last_passage = passage
