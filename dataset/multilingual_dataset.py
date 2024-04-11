import torch
from data_process.data_enum import DatasetEnum, DataTypeEnum, PlmEnum
from data_process.event import EventTuple
from dataset.base_dataset import BaseDataset, unit_test
from tool.repeat_helper import Repeat


class MultilingualDataset(BaseDataset):
    def __init__(self, data_set: DatasetEnum, data_type: DataTypeEnum, plm_model: PlmEnum, k: int):
        super().__init__(data_set, data_type, plm_model, k)

    def __getitem__(self, idx):
        event_tuple: EventTuple = self.tuples[idx]
        zh_trigger = Repeat(event_tuple.self.zh_trigger) if Repeat(event_tuple.self.zh_trigger) else event_tuple.self.zh_trigger
        es_trigger = Repeat(event_tuple.self.es_trigger) if Repeat(event_tuple.self.es_trigger) else event_tuple.self.es_trigger

        self_encoding = self.encoder.encode_one(f"{event_tuple.self.trigger} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.self.mention}")
        self_encoding2 = self.encoder.encode_one(f"{self.encoder.tokenizer.mask_token} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.self.mention}")
        zh_encoding = self.encoder.encode_one(f"{zh_trigger}触发{self.encoder.tokenizer.mask_token}事件。{self.encoder.tokenizer.sep_token} {event_tuple.self.zh_mention}")
        es_encoding = self.encoder.encode_one(f"{es_trigger} descencadena evento {self.encoder.tokenizer.mask_token}.{self.encoder.tokenizer.sep_token} {event_tuple.self.es_mention}")
        neu_encoding = self.encoder.encode_one(f"{event_tuple.neutral.trigger} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.neutral.mention}")
        neg_encoding = self.encoder.encode_one(f"{event_tuple.negative.trigger} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {event_tuple.negative.mention}")

        tokens = self.encoder.tokenizer.tokenize(event_tuple.self.trigger)
        trigger_idx = torch.tensor(self.encoder.tokenizer.convert_tokens_to_ids(tokens))
        trigger_label = torch.zeros((self.encoder.tokenizer.vocab_size), dtype=torch.float).scatter(0, trigger_idx, 1)

        item = {
            "self_event": self_encoding,
            "self_trigger": self_encoding2,
            "zh_event": zh_encoding,
            "es_event": es_encoding,
            "neu_event": neu_encoding,
            "neg_event": neg_encoding,
            "self_trigger_label": trigger_label,
            "self_event_label": torch.LongTensor([event_tuple.self.type_id]),
        }
        return item

    def __len__(self):
        return len(self.tuples)


def main():
    dataset = MultilingualDataset(DatasetEnum.Ace, DataTypeEnum.Train, PlmEnum.XLM, 4)
    unit_test(dataset)


if __name__ == '__main__':
    main()
