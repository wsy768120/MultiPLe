from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Tuple

MAX_MENTION_LENGTH = 512


@dataclass
class Event:
    doc_id: str
    type: str
    mention: str
    trigger: str
    zh_mention: str
    zh_trigger: str
    es_mention: str
    es_trigger: str
    type_id: Optional[int] = None


def set_type_id(events: List[Event], start_index: int = 0):
    """
    按出现频次的降序为事件类型编号
    """
    type_counts_dec = Counter([e.type for e in events]).most_common()
    event_types = [x[0] for x in type_counts_dec]
    type2id = {t: idx for idx, t in enumerate(event_types)}
    for event in events:
        event.type_id = type2id[event.type] + start_index


@dataclass
class EventDataList:
    train: List[Event]
    dev: List[Event]
    test: List[Event]


@dataclass
class EventData:
    seen: EventDataList
    # unseen: EventDataList


@dataclass
class EventTuple:
    self: Event
    neutral: Event
    negative: Event