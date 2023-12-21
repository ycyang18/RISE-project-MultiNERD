import os
import pandas as pd
import torch
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.config import *
from src.utils  import *

def load_tokenizer(model_name: str='bert-base-uncased') -> PreTrainedTokenizer:
    #model_path = os.path.join(MODEL, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

class MultiNERDDataset(Dataset):
    def __init__(self, lang='en', split='train', label_set=ALL_TAGS, max_len=64, model_name='bert-base-uncased'):
        """
        
        Parameters:
        lang (str): Language code.
        split (str): Dataset split, 'train', 'val', or 'test'.
        label_set (dict): Mapping from label names to indices.
        max_len (int): Maximum sequence length.
        model_name (str): Model name for the tokenizer.
        """
        self.dataset = load_jsonl_to_dict(os.path.join(DATA, lang, f'{split}.jsonl'))
        self.max_len = max_len
        self.label_set = label_set
        self.idx2label = {ALL_TAGS[k]: k for k in ALL_TAGS}
        self.tokenizer = load_tokenizer(model_name)
        self.original_tokens = self.dataset['tokens'].tolist()
        self.original_labels = self.dataset['ner_tags'].tolist()
        self.tokenized_tokens = self.tokenize(self.original_tokens)
        self.aligned_labels, self.aligned_tokens = self.align_labels(self.tokenized_tokens, self.original_labels)

    def tokenize(self, original_tokens):
        tokenized_tokens = []
        for tokens in tqdm(original_tokens):
            input_ids = [(self.tokenizer.encode(token, add_special_tokens=False)) for token in tokens]
            tokenized_tokens.append(input_ids)
        return tokenized_tokens
    
    def align_labels(self, tokenized_tokens, original_labels):
        assert len(tokenized_tokens) == len(original_labels)
        aligned_labels = []
        aligned_tokens = []
        for tokens, labels in zip(tokenized_tokens, original_labels):
            assert len(tokens) == len(labels)
            aligned_label = []
            aligned_token = []
            for token, label in zip(tokens, labels):
                tag = self.idx2label[label]
                if tag not in self.label_set:
                    label = 0
                else:
                    label = self.label_set[tag]
                for t in token:
                    aligned_token.append(t)
                    aligned_label.append(label)
            assert len(aligned_token) == len(aligned_label)
            aligned_tokens.append(aligned_token)
            aligned_labels.append(aligned_label)
        return aligned_labels, aligned_tokens

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_ids = self.aligned_tokens[idx]
        labels = self.aligned_labels[idx]
        if len(input_ids) >= self.max_len - 2:
            input_ids = input_ids[:self.max_len - 2]
        if len(labels) >= self.max_len - 2:
            labels = labels[:self.max_len - 2]
        input_ids = [101] + input_ids + [102]
        labels = [0] + labels + [0] #[-100] + labels + [-100]
        ids_len = len(input_ids)
        input_ids = pad_id(input_ids, max_len=self.max_len)
        labels = pad_id(labels, max_len=self.max_len, padding_token=-100)
        masks = pad_id(ids_len * [1], max_len=self.max_len)
        return [
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        ]

class MultiNERDMultiTaskDataset(MultiNERDDataset):
    def __init__(self, lang='en', split='train', label_set=ALL_TAGS, max_len=64, model_name='bert-base-uncased'):
        MultiNERDDataset.__init__(self, lang, split, label_set, max_len, model_name)
        self.rev_label_set = {label_set[k]: k for k in label_set} # id -> tag
        if self.label_set == ALL_TAGS:
            self.b_mapping = ALL_TAGS_B_BINARY
            self.i_mapping = ALL_TAGS_I_BINARY
            self.o_mapping = ALL_TAGS_O_BINARY
        else:
            self.b_mapping = SELECTED_TAGS_B_BINARY
            self.i_mapping = SELECTED_TAGS_I_BINARY
            self.o_mapping = SELECTED_TAGS_O_BINARY
        self.b_labels, self.i_labels, self.o_labels = self.get_multi_task_labels(self.aligned_labels)

    def get_multi_task_labels(self, aligned_labels):
        all_b_labels, all_i_labels, all_o_labels = [], [], []
        for labels in aligned_labels:
            tags = [self.rev_label_set[l] for l in labels]
            b_labels = [self.b_mapping[t] for t in tags]
            i_labels = [self.i_mapping[t] for t in tags]
            o_labels = [self.o_mapping[t] for t in tags]
            all_b_labels.append(b_labels)
            all_i_labels.append(i_labels)
            all_o_labels.append(o_labels)
        return (
            all_b_labels,
            all_i_labels,
            all_o_labels)

    def __getitem__(self, idx):
        input_ids = self.aligned_tokens[idx]
        labels = self.aligned_labels[idx]
        b_labels = self.b_labels[idx]
        i_labels = self.i_labels[idx]
        o_labels = self.o_labels[idx]
        if len(input_ids) >= self.max_len - 2:
            input_ids = input_ids[:self.max_len - 2]
        if len(labels) >= self.max_len - 2:
            labels = labels[:self.max_len - 2]
        input_ids = [101] + input_ids + [102]
        labels = [0] + labels + [0]
        b_labels = [0] + b_labels + [0]
        i_labels = [0] + i_labels + [0]
        o_labels = [0] + o_labels + [0]
        ids_len = len(input_ids)
        input_ids = pad_id(input_ids, max_len=self.max_len)
        labels = pad_id(labels, max_len=self.max_len, padding_token=-100)
        b_labels = pad_id(b_labels, max_len=self.max_len, padding_token=-100)
        i_labels = pad_id(i_labels, max_len=self.max_len, padding_token=-100)
        o_labels = pad_id(o_labels, max_len=self.max_len, padding_token=-100)
        masks = pad_id(ids_len * [1], max_len=self.max_len)
        return [
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
            torch.tensor(b_labels, dtype=torch.long),
            torch.tensor(i_labels, dtype=torch.long),
            torch.tensor(o_labels, dtype=torch.long),
        ]

def prepare_loader(dataset, batch_size):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def prepare_trainer(model, trainloader, epochs, lr):
    print("Loading optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8, no_deprecation_warning=True)
    steps = len(trainloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=steps)
    return optimizer, scheduler