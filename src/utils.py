import os
import json
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from typing import List, Union
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from src.config import *

def load_jsonl(input_path: str) -> List[dict]:
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def jsonl_to_dict(jsonl: List[dict]) -> dict:
    result = {}
    for d in jsonl:
        keys = list(d.keys())
        for k in keys:
            if k in result:
                result[k].append(d[k])
            else:
                result[k] = [d[k]]
    return result

def load_jsonl_to_dict(input_path: str) -> pd.DataFrame:
    """
    Loads a JSONL file and converts it into a pandas DataFrame.

    Parameters:
    input_path (str): The path to the JSONL file.

    Returns:
    pd.DataFrame: A pandas DataFrame created from the JSONL file.
    """
    result = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            for key, value in data.items():
                result.setdefault(key, []).append(value)

    return pd.DataFrame.from_dict(result)

def pad_id(ids, max_len, padding_token=0):
    if len(ids) > max_len:
        return ids[:max_len]
    padding_len = max_len - len(ids)
    ids = ids + [padding_token] * padding_len
    return ids

def seed_all(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def align_show_in_terminal(
        *inputs: List[List[Union[int, str]]],
        header: Union[str, List[str]]=None,
        column_width: Union[int, List[int], str]='auto',
        sep: str='',
        truncate: bool=True,
        truncated_size: int=10,
        truncated_pad_token: str="...") -> None:
    num_cols = len(inputs)
    inputs = [[str(i) for i in inp] for inp in inputs]
    if header:
        if isinstance(header, str):
            header = [header] * num_cols
        else:
            assert len(header) == num_cols
        inputs = [[header[i]] + inp for i, inp in enumerate(inputs)]
    if isinstance(column_width, list):
        assert num_cols == len(column_width), f"Column size mismatched, {num_cols} != {len(column_width)}."
    elif isinstance(column_width, int):
        column_width = [column_width] * num_cols
    elif isinstance(column_width, str):
        column_width = [max(len(i) for i in inp) + 5 for inp in inputs]
    if truncate:
        inputs = [col[:truncated_size] for col in inputs]
        inputs = [col + [truncated_pad_token] for col in inputs]
    aligned_inputs = []
    for size, inp in zip(column_width, inputs):
        aligned_inputs.append([i + " " * (size - len(i)) for i in inp])
    num_rows = len(aligned_inputs[0])
    for i in range(num_rows):
        print(sep.join([aligned_inputs[j][i] for j in range(num_cols)]))

def present_args(args):
    print("=" * 30)
    print("Settings:")
    align_show_in_terminal(
        [i + ":" for i in args.__dict__.keys()],
        args.__dict__.values(),
        truncate=False)
    print("=" * 30)

def mkdir(f_path):
    if os.path.exists(f_path):
        print(f"{f_path} already exists.")
    else:
        os.mkdir(f_path)
    return f_path

def present(dict_obj, indent=4, dst=None):
    json_str = json.dumps(dict_obj, indent=indent)
    if dst is not None:
        with open(dst, 'wt') as writer:
            writer.write(json_str)
    else:
        print(json_str)

def save_exp_results(logs: pd.DataFrame, model: nn.Module, folder_path: str, save_model=False):
    mkdir(folder_path)
    log_path = os.path.join(folder_path, 'logs.tsv')
    logs.to_csv(log_path, sep='\t', index=False)
    if save_model:
        model_path = os.path.join(folder_path, 'model.pkl')
        torch.save(model.state_dict(), model_path)

def load_model(model: nn.Module, state_folder: str):
    state_path = os.path.join(state_folder, 'model.pkl')
    model.load_state_dict(torch.load(state_path))
    return model

def save_json(data: dict, file_path:str, indent: int=4):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(path: str) -> dict:
    with open(path) as f:
        dict_obj = json.load(f)
    return dict_obj

@torch.no_grad()
def ner_inference(model: nn.Module, dataloader, device):
    model.eval()
    all_labels, prd_labels = [], []
    total_loss = 0
    for batch in dataloader:
        batch_ids, batch_labels, batch_masks = [t.to(device) for t in batch]
        outputs = model(batch_ids, attention_mask=batch_masks, labels=batch_labels)
        loss: torch.Tensor = outputs[0]
        total_loss += loss.item()
        logits: torch.Tensor = outputs[1]
        prd_ids: torch.Tensor = logits.argmax(-1) # (batch_size, max_len)
        all_labels.append(batch_labels.to('cpu'))
        prd_labels.append(prd_ids.to('cpu'))
    all_labels = torch.cat(all_labels, dim=0)
    prd_labels = torch.cat(prd_labels, dim=0)
    avg_loss = total_loss / len(dataloader)
    return prd_labels, all_labels, avg_loss

def eval(prd_labels: torch.Tensor, all_labels: torch.Tensor):
    prd_labels = prd_labels.tolist()
    all_labels = all_labels.tolist()
    preds, trues = [], []
    sentence_level_preds, sentence_level_trues = [], []
    assert len(prd_labels) == len(all_labels)
    for i in range(len(all_labels)):
        # exclude padding
        prd_label, all_label = prd_labels[i], all_labels[i]
        prd_label, all_label = prd_label[1:-1], all_label[1:-1]
        all_label = [l for l in all_label if l != -100]
        prd_label = prd_label[:len(all_label)]
        preds += prd_label
        trues += all_label
        sentence_level_preds.append(prd_label)
        sentence_level_trues.append(all_label)

    # Calculate metrics
    macro_f1, micro_f1 = f1_score(trues, preds, average='macro'), f1_score(trues, preds, average='micro')
    accuracy = accuracy_score(trues, preds)
    precision, recall, _, _ = precision_recall_fscore_support(trues, preds, average='macro')
    metrics_result = {
        'accuracy(micro_f1)': accuracy, # Do not consider label imbalance
        'macro_f1': macro_f1, # Consider label imbalance
        'precision': precision,
        'recall':recall
    }
    return metrics_result, sentence_level_preds, sentence_level_trues