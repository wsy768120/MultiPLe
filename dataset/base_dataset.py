import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BatchEncoding
from data_process.data_enum import DatasetEnum, DataTypeEnum, PlmEnum
from data_process.event import Event, EventTuple
from data_process.gen_train_data import load_event_tuples
from tool import pretrain_model_helper
from tool.log_helper import logger


class BaseDataset(Dataset):
    def __init__(self, data_set: DatasetEnum, data_type: DataTypeEnum, plm_model: PlmEnum, k: int):
        self.data_set = data_set
        self.data_type = data_type
        self.encoder = EventMentionEncoder.get_encoder(plm_model)
        self.tuples = load_event_tuples(data_set, data_type, k)
        self.k = k

    def __getitem__(self, idx):
        event_tuple: EventTuple = self.tuples[idx]
        self_encoding = self.encoder.encode_one(f"{event_tuple.self.trigger} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.self.mention}")
        self_encoding2 = self.encoder.encode_one(f"{self.encoder.tokenizer.mask_token} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.self.mention}")
        neu_encoding = self.encoder.encode_one(f"{event_tuple.neutral.trigger} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.neutral.mention}")
        neg_encoding = self.encoder.encode_one(f"{event_tuple.negative.trigger} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.negative.mention}")

        tokens = self.encoder.tokenizer.tokenize(event_tuple.self.trigger)
        trigger_idx = torch.tensor(self.encoder.tokenizer.convert_tokens_to_ids(tokens))
        trigger_label = torch.zeros((self.encoder.tokenizer.vocab_size), dtype=torch.float).scatter(0, trigger_idx, 1)
        item = {
            "self_event": self_encoding,
            "self_trigger": self_encoding2,
            "neu_event": neu_encoding,
            "neg_event": neg_encoding,
            "self_trigger_label": trigger_label,
            "self_event_label": torch.LongTensor([event_tuple.self.type_id]),
        }
        return item

    def __len__(self):
        return len(self.tuples)


ENCODE_MAX_LENGTH = 100
ENCODER_MAP = dict()


class EventMentionEncoder:

    @staticmethod
    def get_encoder(plm_model: PlmEnum):
        """
        线程不安全！！！！！！
        """
        if plm_model not in ENCODER_MAP:
            ENCODER_MAP[plm_model] = EventMentionEncoder(plm_model)
        return ENCODER_MAP[plm_model]

    def __init__(self, plm_model: PlmEnum):
        self.tokenizer = pretrain_model_helper.load_tokenizer(plm_model)

    def encode_one(self, sentence: str) -> BatchEncoding:
        plm_encoding = self.tokenizer(sentence, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        for k, v in plm_encoding.data.items():
            plm_encoding[k] = v[0]
        return plm_encoding

    def encode_pair_trigger(self, event: Event, pattern: str) -> BatchEncoding:
        event_ontology = f"{event.mention}{self.tokenizer.sep_token} Trigger word: a word that can trigger an event, usually a verb or noun in the sentence."
        plm_encoding = self.tokenizer(pattern, event_ontology, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        # ['[CLS]', 'trigger', 'word', 'is', '[MASK]', '.', '[SEP]', 'after', 'an', 'education', 'in', 'exclusive', 'california', '##n', 'private', 'schools', ',', 'her', 'theatrical', 'debut', 'was', 'with', 'her', 'mother', 'in', '"', 'lil', '##iom', '"', ',', 'a', 'play', 'produced', 'by', 'the', 'so', '##mbre', '##ro', 'theater', ',', 'in', 'phoenix', ',', 'arizona', ',', 'in', 'april', '1955', '[SEP]']
        for k, v in plm_encoding.data.items():
            plm_encoding[k] = v[0]
        return plm_encoding

    def encode_pair_trigger_sequence(self, event: Event, pattern: str) -> BatchEncoding:
        event_ontology = f"Trigger word: a word that can trigger an event, usually a verb or noun in the sentence.{self.tokenizer.sep_token} {event.mention}"
        plm_encoding = self.tokenizer(pattern, event_ontology, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        # ['[CLS]', 'trigger', 'word', 'is', '[MASK]', '.', '[SEP]', 'after', 'an', 'education', 'in', 'exclusive', 'california', '##n', 'private', 'schools', ',', 'her', 'theatrical', 'debut', 'was', 'with', 'her', 'mother', 'in', '"', 'lil', '##iom', '"', ',', 'a', 'play', 'produced', 'by', 'the', 'so', '##mbre', '##ro', 'theater', ',', 'in', 'phoenix', ',', 'arizona', ',', 'in', 'april', '1955', '[SEP]']
        for k, v in plm_encoding.data.items():
            plm_encoding[k] = v[0]
        return plm_encoding

    def encode_pair_event_woot(self, event: Event, pattern: str) -> BatchEncoding:
        trigger = event.trigger.split()
        event_ontology = f"{event.mention} {self.tokenizer.sep_token} Trigger word is {trigger[0]}. "
        plm_encoding = self.tokenizer(pattern, event_ontology, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        for k, v in plm_encoding.data.items():
            plm_encoding[k] = v[0]
        return plm_encoding

    def encode_pair_event(self, event: Event, pattern: str) -> BatchEncoding:
        trigger = event.trigger.split()
        event_ontology = f"{event.mention} {self.tokenizer.sep_token} Event: which type the sentence or trigger belongs to.{self.tokenizer.sep_token} Trigger word is {trigger[0]}."
        # event sequence:
        # event_ontology = f"{event.mention} {self.tokenizer.sep_token} Trigger word is {trigger[0]}. {self.tokenizer.sep_token} Event: which type the sentence or trigger belongs to."
        # event_ontology = f"Event: which type the sentence or trigger belongs to. {self.tokenizer.sep_token} {event.mention} {self.tokenizer.sep_token} Trigger word is {trigger[0]}."
        # event_ontology = f"Event: which type the sentence or trigger belongs to. {self.tokenizer.sep_token} Trigger word is {trigger[0]}. {self.tokenizer.sep_token} {event.mention}"
        # event_ontology = f"Trigger word is {trigger[0]}. {self.tokenizer.sep_token} {event.mention} {self.tokenizer.sep_token} Event: which type the sentence or trigger belongs to."
        # event_ontology = f"Trigger word is {trigger[0]}. {self.tokenizer.sep_token} Event: which type the sentence or trigger belongs to. {self.tokenizer.sep_token} {event.mention}"
        # change template:
        # event_ontology = f"{event.mention} {self.tokenizer.sep_token} Event: what is included in the sentence.{self.tokenizer.sep_token} Trigger word is {trigger[0]}."
        plm_encoding = self.tokenizer(pattern, event_ontology, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        for k, v in plm_encoding.data.items():
            plm_encoding[k] = v[0]
        return plm_encoding

    def encode_pair(self, event: Event, pattern: str) -> BatchEncoding:
        plm_encoding = self.tokenizer(pattern, event.mention, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        # ['[CLS]', 'this', 'is', 'event', 'about', '[MASK]', '.', '[SEP]'after', 'an', 'education', 'in', 'exclusive', 'california', '##n', 'private', 'schools', ',', 'her', 'theatrical', 'debut', 'was', 'with', 'her', 'mother', 'in', '"', 'lil', '##iom', '"', ',', 'a', 'play', 'produced', 'by', 'the', 'so', '##mbre', '##ro', 'theater', ',', 'in', 'phoenix', ',', 'arizona', ',', 'in', 'april', '1955', '[SEP]']
        for k, v in plm_encoding.data.items():
            plm_encoding[k] = v[0]
        return plm_encoding


def unit_test(dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for k, v in next(iter(dataloader)).items():
        if isinstance(v, dict):
            for ki, vi in v.items():
                logger.info(f"{k}.{ki}={vi.shape}")
        else:
            logger.info(f"{k}={v.shape}")
    for _ in tqdm(dataloader):
        pass


def main():
    encoder = EventMentionEncoder.get_encoder(PlmEnum.Bert)
    tuples = load_event_tuples(DatasetEnum.Ace, DataTypeEnum.Train, 4)
    encodings = encoder.encode_one(tuples[0].self.mention)
    encodings = encoder.encode_pair(tuples[0].self, "This is news about [MASK].")
    pass


if __name__ == '__main__':
    main()
