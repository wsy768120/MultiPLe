from typing import List
from torch.utils.data import Dataset
from data_process.event import Event
from dataset.base_dataset import EventMentionEncoder


class BaseValidatorDataset(Dataset):
    def __init__(self, events: List[Event], encoder: EventMentionEncoder):
        self.events = events
        self.encoder = encoder

    def __getitem__(self, idx):
        self_encoding = self.encoder.encode_one(f"{self.encoder.tokenizer.mask_token} triggers {self.encoder.tokenizer.mask_token} event.{self.encoder.tokenizer.sep_token} {self.events[idx].mention}")

        item = {
            "self_event": self_encoding,
        }
        return item

    def __len__(self):
        return len(self.events)
