from data_process.data_enum import PlmEnum
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tool.path_helper import ROOT_DIR

MODEL_DIR = "pretrain/plm/%s"
UNUSED_WORD_COUNT = 999


def load_tokenizer(plm: PlmEnum):
    path = ROOT_DIR.joinpath(MODEL_DIR % plm.value).__str__()
    if plm.value == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
    elif plm.value == "xlm-roberta-large":
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    else:  #"bert-base-multilingual-cased"
        tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)

    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused%d]" % (i + 1) for i in range(UNUSED_WORD_COUNT)]})
    return tokenizer


def load_model(plm: PlmEnum):
    path = ROOT_DIR.joinpath(MODEL_DIR % plm.value).__str__()
    if plm.value == "bert-base-uncased":
        model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
    elif plm.value == "xlm-roberta-large":
        model = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True)
    else:  #"bert-base-multilingual-cased"
        model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
    return model