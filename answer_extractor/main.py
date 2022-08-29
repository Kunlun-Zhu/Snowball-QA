import jsonlines
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from model import Bert, BertBMT
from trainer import Trainer
from dataset import BertIterableDataset, PertIterableDataset
from utils import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import boolean_string
from collections import OrderedDict
# import bmtrain as bmt
import json

torch.manual_seed(1)


parser = argparse.ArgumentParser()

parser.add_argument("--max_seq_len", default=512, type=int, required=True, help="sequence length.")
parser.add_argument("--epochs", default=5, type=int, required=True, help="epoch.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=16, type=int, required=True, help="batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=float, required=True, help="lr.")
parser.add_argument("--weight_decay", default=0.0001, type=float, required=True, help="lr.")
parser.add_argument("--save_interval", default=10, type=int, required=True, help="ckpt nums")
parser.add_argument("--print_interval", default=50, type=int, required=True, help="ckpt nums")
parser.add_argument("--save_path", default='/data/private/wanghuadong/liangshihao/QA/output/', type=str, required=True, help="save directory.")
parser.add_argument("--train_data_dir", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str, required=True, help="save directory.")
parser.add_argument("--test_data_dir", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str, required=True, help="save directory.")
# parser.add_argument("--dataset", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str, required=True, help="save directory.")
parser.add_argument("--pretrained_path", default='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base-multilingual-cased/', type=str, required=True, help="pretrained path.")
parser.add_argument("--gpu_ids", default='6', type=str, required=True, help="gpu ids.")
parser.add_argument("--do_train", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--do_predict", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--eval_model_path", default='', type=str, required=True, help="gpu or cpu.")
parser.add_argument("--inference_path", default='', type=str, required=True, help="gpu or cpu.")

args = parser.parse_args()

bmtrain = False

def build_dataset(dataset, do_train, do_predict):
    # # test data from full data
    # full_dataset = BaseDataset(data_dir=args.data_dir, dataset=dataset, do_train=True, do_eval=True, do_test=False, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
    # train_size = int(0.9 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # test data from apart
    if do_train:
        train_dataset = BaseDataset(data_dir=args.data_dir, dataset=dataset, do_train=True, do_eval=True, do_test=False, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
    test_dataset = BaseDataset(data_dir=args.data_dir, dataset=dataset, do_train=False, do_eval=True, do_test=False, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
    
    if do_predict:
        return test_dataset
    else:
        return train_dataset, test_dataset
# build dataloader
def build_dataloader(train_dataset, test_dataset):

    if bmtrain:
        train_dataloader = DistributedDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DistributedDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def validation(model, test_dataloader, device):
    model.eval()
    total_loss = 0.0
    total_span_loss, total_span_acc, total_zero_loss, total_zero_acc = 0.0, 0.0, 0.0, 0.0 
    for batch_dict in tqdm(test_dataloader):
        with torch.no_grad():
            src_ids, label_ids, input_mask = batch_dict["input_ids"], batch_dict["label_ids"], batch_dict["input_mask"] 
            src_ids, label_ids, input_mask = src_ids.to(device), label_ids.to(device), input_mask.to(device)
            span_loss, zero_loss, span_logits, zero_logits, _ = model(src_ids, label_ids, input_mask)
            # print(logits)
            # print(label_ids[span_pos].view(-1))
            span_pos = (label_ids == 1)
            zero_pos = (label_ids == 0)
            span_acc = span_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[span_pos].view(-1)).sum()
            span_acc = (span_acc * 100 / label_ids[span_pos].view(-1).size(0))
            zero_acc = zero_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[zero_pos].view(-1)).sum()
            zero_acc = (zero_acc * 100 / label_ids[zero_pos].view(-1).size(0))
            total_loss += span_loss.mean()
            total_span_loss += span_loss.mean()
            total_span_acc += span_acc
            total_zero_acc += zero_acc
            total_zero_loss += zero_loss.mean()
    total_test_data = len(test_dataloader)
    span_loss = total_span_loss / total_test_data
    zero_loss = total_zero_loss / total_test_data
    span_acc = total_span_acc / total_test_data
    zero_acc = total_zero_acc / total_test_data
    print("Validation. span_loss:{}, span_acc:{}, zero_loss:{}, zero_acc:{}".format
        (str(span_loss.cpu().detach().numpy()), str(span_acc.cpu().detach().numpy()), str(zero_loss.cpu().detach().numpy()), str(zero_acc.cpu().detach().numpy()))
    )
    model.train()
    return total_loss / total_test_data

def train(model, train_dataloader, test_dataloader, device):
    model.train()
    # build optim
    optimizer = optimizer = optim.Adam(get_model_obj(model).parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    step_per_epoch = len(train_dataloader)
    # train progress
    total_train_step = step_per_epoch * args.epochs
    save_interval = total_train_step / args.save_interval
    print_interval = total_train_step / args.print_interval
    last_save_val_loss = 100.0
    for epoch in range(args.epochs):
        for iter, batch_dict in enumerate(train_dataloader):
            src_ids, label_ids, input_mask = batch_dict["input_ids"], batch_dict["label_ids"], batch_dict["input_mask"] 
            src_ids, label_ids, input_mask = src_ids.to(device), label_ids.to(device), input_mask.to(device)
            with torch.no_grad():
                span_pos = (label_ids == 1)
                zero_pos = (label_ids == 0)
            optimizer.zero_grad()
            span_loss, zero_loss, span_logits, zero_logits, _ = model(src_ids, label_ids, input_mask)
            # print(logits)
            # print(label_ids[span_pos].view(-1))
            span_acc = span_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[span_pos].view(-1)).sum()
            span_acc = (span_acc * 100 / label_ids[span_pos].view(-1).size(0))
            zero_acc = zero_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[zero_pos].view(-1)).sum()
            zero_acc = (zero_acc * 100 / label_ids[zero_pos].view(-1).size(0))
            loss = zero_loss.mean() + span_loss.mean()
            loss.backward()
            optimizer.step()
            if iter % int(print_interval) == 0:
                print("Step:{}/{}, span_loss:{}, span_acc:{}, zero_loss:{}, zero_acc:{}".format
                    (str(iter), str(total_train_step), str(loss.cpu().detach().numpy()), str(span_acc.cpu().detach().numpy()), str(zero_loss.cpu().detach().numpy()), str(zero_acc.cpu().detach().numpy()))
                )
            if iter % int(save_interval) == 0:
                val_loss = validation(model, test_dataloader, device)
                if last_save_val_loss > val_loss:
                    last_save_val_loss = val_loss
                    torch.save(get_model_obj(model).state_dict(), args.save_path + 'epoch_{}.pt'.format(str(epoch)))
        scheduler.step()

def inference(model, test_dataloader, tokenizer, inference_path, device):
    model.eval()
    total_loss = 0.0
    total_span_loss, total_span_acc, total_zero_loss, total_zero_acc = 0.0, 0.0, 0.0, 0.0
    output_dict_list = []
    min_span_len = 2
    with jsonlines.open(inference_path, mode='w') as writer:
        writer._flush = True
        for batch_dict in tqdm(test_dataloader):
            with torch.no_grad():
                src_ids, label_ids, input_mask = batch_dict["input_ids"], batch_dict["label_ids"], batch_dict["input_mask"] 
                src_ids, label_ids, input_mask = src_ids.to(device), label_ids.to(device), input_mask.to(device)
                span_loss, zero_loss, span_logits, zero_logits, logits = model(src_ids, label_ids, input_mask)
                # print(logits)
                # print(label_ids[span_pos].view(-1))
                span_pos = (label_ids == 1)
                zero_pos = (label_ids == 0)
                zero_pos = torch.logical_and(zero_pos, (src_ids != 0))
                
                span_logits = span_logits.view(-1, 2).max(dim=1)[1]
                span_acc = span_logits.eq(label_ids[span_pos].view(-1)).sum()
                span_acc = (span_acc * 100 / label_ids[span_pos].view(-1).size(0))
                zero_acc = zero_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[zero_pos].view(-1)).sum()
                zero_acc = (zero_acc * 100 / label_ids[zero_pos].view(-1).size(0))
                total_loss += span_loss
                total_span_loss += span_loss
                total_span_acc += span_acc
                total_zero_acc += zero_acc
                total_zero_loss += zero_loss

                logits01 = torch.argmax(logits, -1)

                src_ids = src_ids.cpu().detach().numpy()
                logits01 = logits01.cpu().detach().numpy()
                logits = logits.cpu().detach().numpy()
                label_ids = label_ids.cpu().detach().numpy()
                
                for idx, (src, logit) in enumerate(zip(src_ids, logits01)):
                    output_dict = extract_answer(tokenizer, src, logit, label_ids[idx], min_span_len, logits[idx])
                    passage = output_dict["passage"]
                    for candi_ans_dict in output_dict["candidate_answers"]:
                        answer_text = candi_ans_dict["answer_text"]
                        answer_start = candi_ans_dict["answer_start"]
                        writer.write({"passage":passage, "answer":answer_text, "answer_start":int(answer_start), "question":'', "language":'zh'})
                        # writer.flush()
    total_test_data = len(test_dataloader)
    span_loss = total_span_loss / total_test_data
    zero_loss = total_zero_loss / total_test_data
    span_acc = total_span_acc / total_test_data
    zero_acc = total_zero_acc / total_test_data
    print("Validation. span_loss:{}, span_acc:{}, zero_loss:{}, zero_acc:{}".format
        (str(span_loss.cpu().detach().numpy()), str(span_acc.cpu().detach().numpy()), str(zero_loss.cpu().detach().numpy()), str(zero_acc.cpu().detach().numpy()))
    )
    model.train()
    return total_loss / total_test_data
def main():
    # build model
    cudnn.benchmark = True
    # torch.cuda.set_device(3)
    device_ids = list()
    for gpu_id in args.gpu_ids:
        device_ids.append(int(gpu_id))
    device = torch.device('cuda:{}'.format(device_ids[0]) if args.use_cuda else 'cpu')

    model_config = {"pretrained_path": args.pretrained_path, "device_ids": device_ids}

    if bmtrain:
        import bmtrain as bmt
        bmt.init_distributed(seed=0)
        model = BertBMT(model_config=model_config)
        bmt.init_parameters(model)
        # model.to(device) 
    else:
        model = Bert(model_config=model_config)
        model.to(device) 
    # build dataloader
    if args.do_train:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
        train_dataset = BertIterableDataset(args.train_data_dir, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
        test_dataset = BertIterableDataset(args.test_data_dir, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        # train_dataloader, test_dataloader = build_dataloader(train_dataset, test_dataset)
        train(model, train_dataloader, test_dataloader, device)
    if args.do_predict:
        state_dict = torch.load(args.eval_model_path, map_location=device)
        model.load_state_dict(state_dict)
        test_dataset = BertIterableDataset(args.test_data_dir, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
        inference(model, test_dataloader, test_dataset.get_tokenizer(), args.inference_path, device)

if __name__ == '__main__':
    main()




