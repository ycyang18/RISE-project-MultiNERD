import os
import sys

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC   = os.path.join(ROOT, 'src')
DATA  = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')
LOG   = os.path.join(ROOT, 'log')
RESULT = os.path.join(ROOT, 'result')
sys.path.append(ROOT)

ALL_TAGS = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
}

SELECTED_TAGS = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-DIS': 7,
    'I-DIS': 8,
    'B-ANIM': 9,
    'I-ANIM': 10
    }

ALL_TAGS_B_BINARY = {tag: int("B-" in tag) for tag in ALL_TAGS}
ALL_TAGS_I_BINARY = {tag: int("I-" in tag) for tag in ALL_TAGS}
ALL_TAGS_O_BINARY = {tag: int(tag == "O")  for tag in ALL_TAGS}

SELECTED_TAGS_B_BINARY = {tag: int("B-" in tag) for tag in SELECTED_TAGS}
SELECTED_TAGS_I_BINARY = {tag: int("I-" in tag) for tag in SELECTED_TAGS}
SELECTED_TAGS_O_BINARY = {tag: int(tag == "O")  for tag in SELECTED_TAGS}

BERT_PATH = os.path.join(MODEL, 'bert-base-uncased')
RoBERTa_PATH = os.path.join(MODEL, 'roberta-base')