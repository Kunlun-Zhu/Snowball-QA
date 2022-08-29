import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from model_center.layer import Embedding, Linear, LayerNorm, Encoder
import bmtrain as bmt
# from model_center.model import Bert, BertConfig

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        selected_token_tensor = hidden_states
        pooled_output = self.dense(selected_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Bert(nn.Module):
    def __init__(self, model_config: argparse.Namespace):
        super().__init__()
        self.pretrained_model_path = model_config['pretrained_path']
        self.config = BertConfig.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_path)
        self.encoder = BertModel.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_path)
        self.pooler = BertPooler(self.config)
        self.num_labels = 2
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        # self.pooler = torch.nn.DataParallel(self.pooler, device_ids=model_config["device_ids"])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)

    def forward(self, input_ids, labels, input_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        with torch.no_grad():
            span_pos = (labels == 1)
            zero_pos = (labels == 0)
            # zero_pos = torch.logical_and(zero_pos, (input_ids != 0))

        logits = self.pooler(hidden_states)
        logits = self.classifier(logits)
        span_logits = logits[span_pos]
        zero_logits = logits[zero_pos]
        if self.num_labels == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
            loss_fct = nn.CrossEntropyLoss()
            span_loss = loss_fct(span_logits.view(-1, self.num_labels), labels[span_pos].view(-1))
            zero_loss = loss_fct(zero_logits.view(-1, self.num_labels), labels[zero_pos].view(-1))
        return span_loss, zero_loss, span_logits, zero_logits, logits

class BertBMT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # config = BertConfig.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        # self.bert = Bert.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.encoder = Bert.from_pretrained("bert-base-uncased")
        self.pooler = BertPooler(self.config)
    def forward(self, input_ids, labels, input_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        with torch.no_grad():
            span_pos = (labels == 1)
            zero_pos = (labels == 0)

        logits = self.pooler(hidden_states)
        logits = self.classifier(logits)
        span_logits = logits[span_pos]
        zero_logits = logits[zero_pos]
        if self.num_labels == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
            loss_fct = nn.CrossEntropyLoss()
            span_loss = loss_fct(span_logits.view(-1, self.num_labels), labels[span_pos].view(-1))
            zero_loss = loss_fct(zero_logits.view(-1, self.num_labels), labels[zero_pos].view(-1))
        return span_loss, zero_loss, span_logits, zero_logits