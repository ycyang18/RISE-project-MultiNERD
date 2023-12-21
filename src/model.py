import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from src.config import *

def token_wise_CELoss(pooled_logits: torch.Tensor, labels: torch.Tensor, num_labels: int):
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(pooled_logits.view(-1, num_labels), labels.view(-1))
    return loss

class BertNEROriginal(nn.Module):
    def __init__(self, num_labels):
        super(BertNEROriginal, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.predict = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None): # input_ids: [batch_size, length]
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.predict(sequence_output) # (B, N) -> (B, 2)

        loss = None
        if labels is not None:
            loss = token_wise_CELoss(logits.view(-1, self.num_labels), labels.view(-1), self.num_labels)
        return loss, logits

class RoBERTaNEROriginal(nn.Module):
    def __init__(self, num_labels):
        super(RoBERTaNEROriginal, self).__init__()
        self.num_labels = num_labels
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.predict = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None): # input_ids: [batch_size, length]
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.predict(sequence_output)

        loss = None
        if labels is not None:
            loss = token_wise_CELoss(logits.view(-1, self.num_labels), labels.view(-1), self.num_labels)
        return loss, logits

class BertNERLSTM(nn.Module):
    def __init__(self, num_labels, hidden_size=1024):
        super(BertNERLSTM, self).__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            768, self.hidden_size,
            bias=True,
            bidirectional=True,
            batch_first=True)
        self.predict = nn.Linear(2 * self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_outputs, (_, _) = self.lstm(sequence_output) # [B, L, 2 * H]
        logits = self.predict(lstm_outputs) # [B, L, num_labels]

        loss = None
        if labels is not None:
            loss = token_wise_CELoss(logits.view(-1, self.num_labels), labels.view(-1), self.num_labels)
        return loss, logits

ALL_MODELS = {
        'BertNEROriginal': BertNEROriginal,
        'BertNERLSTM': BertNERLSTM,
        'RoBERTaNEROriginal': RoBERTaNEROriginal
    }

MODEL_MAPPING = {
    'BertNEROriginal': 'bert-base-uncased',
    'BertNERLSTM': 'bert-base-uncased',
    'RoBERTaNEROriginal': 'roberta-base'
}

def init_model(model_name, num_label) -> nn.Module:
    model = ALL_MODELS[model_name](num_label)
    return model