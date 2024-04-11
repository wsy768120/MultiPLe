import itertools
import pickle
import random
from typing import List
from tqdm import tqdm
from data_process.data_enum import DatasetEnum, RANDOM_STATE, DataTypeEnum
from data_process.data_reader.main_data_reader import load_events
from data_process.event import Event, EventTuple
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, mkdir_parent

EVENT_TUPLES_PATH = "out/processed_data/%s/%s/%s/events_tuples.pt"


def parent_type(type_id_set: set, data_set: DatasetEnum):
    sorted_type_id_list = sorted(type_id_set, key=lambda x: x[1])
    if data_set.value == 'Ace':
        parent_type_list = [(type_id[0].split(":")[0], type_id[1]) for type_id in sorted_type_id_list]
    if data_set.value == 'FewShotED':
        parent_type_list = [(type_id[0].split(".")[0], type_id[1]) for type_id in sorted_type_id_list]
    parent_list = []
    parent_child_list = []
    for i in parent_type_list:
        if i[0] not in parent_list:
            parent_list.append(i[0])
            parent_child_list.append([i[1]])
        else:
            parent_child_list[parent_list.index(i[0])].append(i[1])
    return parent_child_list, len(parent_list)


def gen_train_tuples(events: List[Event], data_set: DatasetEnum) -> List[EventTuple]:
    events = sorted(events, key=lambda e: e.type)
    onetime_random = random.Random(RANDOM_STATE)
    last_type = ""
    event_tuples = []
    child_type_events = []
    other_type_events = []
    for self_event in tqdm(events, desc="gen_train_tuples", ncols=150):
        if self_event.type != last_type:
            type_id_set = set([(e.type, e.type_id) for e in events])
            parent_child_list, _ = parent_type(type_id_set, data_set)
            for child_list in parent_child_list:
                if self_event.type_id in child_list:
                    child_type_events = [e for e in events if e.type_id == self_event.type_id]  # 同一子标签
                    other_type_events = [e for e in events if (e.type_id in child_list) and (e.type_id != self_event.type_id)]  # 同一父标签，不同子标签
                    other_type_events = [e for e in events if e.type_id not in child_list] if other_type_events == [] else other_type_events  # 不同父标签
                    break
            onetime_random.shuffle(child_type_events)
            onetime_random.shuffle(other_type_events)
            # Return elements from the iterable until it is exhausted. Then repeat the sequence indefinitely.
            child_type_events_ic = itertools.cycle(child_type_events)
            other_type_events_ic = itertools.cycle(other_type_events)
            last_type = self_event.type

        # neutral_event 来自同一子标签的不同事件集合
        neutral_event = next(child_type_events_ic)
        while neutral_event.doc_id == self_event.doc_id and len(child_type_events) > 1:
            neutral_event = next(child_type_events_ic)

        # negative_event 来自不同父标签的事件集合
        negative_event = next(other_type_events_ic)

        event_tuple = EventTuple(
            self=self_event,
            neutral=neutral_event,
            negative=negative_event
        )

        event_tuples.append(event_tuple)

    return event_tuples


def load_event_tuples(data_set: DatasetEnum, data_type: DataTypeEnum, k: int) -> List[EventTuple]:
    data_path = ROOT_DIR.joinpath(EVENT_TUPLES_PATH % (data_set.value, k, data_type.value))
    if not data_path.exists():
        events = load_events(data_set, k)
        if data_type == DataTypeEnum.Train:
            tuples = gen_train_tuples(events.seen.train, data_set)
        elif data_type == DataTypeEnum.Dev:
            tuples = gen_train_tuples(events.seen.dev, data_set)
        elif data_type == DataTypeEnum.Test:
            tuples = gen_train_tuples(events.seen.test, data_set)
        else:
            raise Exception("Unsupported Datatype")

        mkdir_parent(data_path)
        with open(data_path, 'wb') as f:
            pickle.dump(tuples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("pickle.dump() end!!")
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def main():
    for k in [4, 8, 16, 32]:
        for data_set in [
            DatasetEnum.Ace,
            DatasetEnum.FewShotED,
        ]:
            train_tuples = load_event_tuples(data_set, DataTypeEnum.Train, k)
            logger.info(f"Train={len(train_tuples)}")
            print("end!!!")


if __name__ == '__main__':
    main()
