from enum import Enum


RANDOM_STATE = 42
SAMPLE_NUMBER_THRESHOLD = 0
MAX_MENTION_LEN = 300


class LanguageEnum(Enum):
    English = "en"
    Chinese = "zh"
    Spanish = "es"


class DatasetEnum(Enum):

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, lang: LanguageEnum):
        self.lang = lang

    def __str__(self):
        return str(self.value)

    Test = "TestDataset", LanguageEnum.English
    Ace = "Ace", LanguageEnum.English
    FewShotED = "FewShotED", LanguageEnum.English


class DataTypeEnum(Enum):
    def __str__(self):
        return str(self.value)

    Train = "train"
    Test = "test"
    Dev = "dev"


class PlmEnum(Enum):
    def __str__(self):
        return str(self.value)

    Bert = "bert-base-uncased"
    XLM = "xlm-roberta-large"
    BM = "bert-base-multilingual-cased"


def get_all_dataset():
    all_dataset = list(DatasetEnum)
    all_dataset.remove(DatasetEnum.Test)
    return all_dataset
