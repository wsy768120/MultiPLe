from tool.path_helper import ROOT_DIR
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "pretrain/plm/Helsinki-NLP/opus-mt-en-%s"


def translator():
    path1 = ROOT_DIR.joinpath(MODEL_DIR % 'es').__str__()
    model1 = AutoModelForSeq2SeqLM.from_pretrained(path1, local_files_only=True)
    tokenizer1 = AutoTokenizer.from_pretrained(path1, local_files_only=True)
    es_translation = pipeline("translation", model=model1, tokenizer=tokenizer1)
    path2 = ROOT_DIR.joinpath(MODEL_DIR % 'zh').__str__()
    model2 = AutoModelForSeq2SeqLM.from_pretrained(path2, local_files_only=True)
    tokenizer2 = AutoTokenizer.from_pretrained(path2, local_files_only=True)
    zh_translation = pipeline("translation", model=model2, tokenizer=tokenizer2)
    return es_translation, zh_translation


if __name__ == '__main__':
    es_translation, zh_translation = translator()
    es_text = es_translation("This example configuration can provide MT service for en->es and en->fi language pairs.", max_length=100)[0]['translation_text']
    zh_text = zh_translation("This example configuration can provide MT service for en->es and en->fi language pairs.", max_length=100)[0]['translation_text']
    print(es_text, zh_text)
