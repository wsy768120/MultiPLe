import pickle
from collections import Counter
from statistics import stdev, mean
from sklearn.model_selection import train_test_split
import random
from data_process.data_enum import DatasetEnum, RANDOM_STATE, LanguageEnum, DataTypeEnum
from data_process.data_reader import ace_reader, fewshoted_reader
from data_process.event import Event, set_type_id, EventData, EventDataList, EventTuple
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, mkdir_parent

DATASET_READER_MAP = {
    DatasetEnum.Ace: ace_reader,
    DatasetEnum.FewShotED: fewshoted_reader,
}

PROCESSED_DATA_PATH = "out/processed_data/%s/all_events.pt"
SPLITED_DATA_PATH = "out/processed_data/%s/%s/events.pt"


def load_events(data_set: DatasetEnum, k: int) -> EventData:
    if data_set.lang != LanguageEnum.English:
        raise NotImplementedError()

    processed_data_path = ROOT_DIR.joinpath(PROCESSED_DATA_PATH % data_set.value)
    if not processed_data_path.exists():
        all_events = DATASET_READER_MAP[data_set].read_events(data_set)
        set_type_id(all_events)
        mkdir_parent(processed_data_path)
        with open(processed_data_path, 'wb') as f:
            pickle.dump(all_events, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("PROCESSED_DATA_PATH:", processed_data_path)

    splited_data_path = ROOT_DIR.joinpath(SPLITED_DATA_PATH % (data_set.value, k))
    if not splited_data_path.exists():
        with open(processed_data_path, 'rb') as f:
            all_events = pickle.load(f)
        if k < 100:
            train_events, dev_events, test_events = k_train_dev_test_split(all_events, k)
        elif k == 100:
            train_events, dev_events, test_events = train_dev_test_split(all_events, k)
        else:
            train_events, dev_events, test_events = train_dev_nk_test_split(all_events, k)
        mkdir_parent(splited_data_path)
        data_to_save = EventData(
            EventDataList(train_events, dev_events, test_events)
        )
        with open(splited_data_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("SPLITED_DATA_PATH:", splited_data_path)

    with open(splited_data_path, 'rb') as f:
        return pickle.load(f)


def load_event_list(data_set: DatasetEnum, data_type: DataTypeEnum, k: int):
    data_path = ROOT_DIR.joinpath(SPLITED_DATA_PATH % (data_set.value, k))
    with open(data_path, 'rb') as f:
        all_events = pickle.load(f)
        if data_type == DataTypeEnum.Train:
            events = all_events.seen.train
        elif data_type == DataTypeEnum.Dev:
            events = all_events.seen.dev
        else:  # data_type == DataTypeEnum.Test
            events = all_events.seen.test
        return events


def train_dev_test_split(events, k): # 8:1:1
    train_events, test_events = train_test_split(
        events, test_size=0.1, random_state=RANDOM_STATE)
    train_events, dev_events = train_test_split(
        train_events, test_size=1 / 9, random_state=RANDOM_STATE)
    logger.info(f"Train={len(train_events)}, Dev={len(dev_events)}, Test={len(test_events)}")
    return train_events, dev_events, test_events

# wsy: 重写train_sev_test_split(), 变为小样本的划分方式
# 比如16-shot的true few shot, 从每个event type下抽取32 instances
# 从32 shot中随机挑选一半为train dataset，一半为dev，不足32的类别忽略（在experiment setting中说明）
# 如果类别总数是n, 抽取的train dataset共包含n*16个实例, 不建立meta-task, 全部放进预训练模型里训练
# 32-shot后剩下的instances全部作为test
def k_train_dev_test_split(events, k: int):
    types_num = Counter([t.type for t in events]).most_common()
    train_events, dev_events, test_events = [], [], []
    idx = 0
    while idx < len(events):
        for item in types_num:
            if item[0] == events[idx].type:
                event_num = item[1]
                break
        if event_num >= k*2:
            random.seed(RANDOM_STATE)
            events_origin = events[idx:idx+event_num]
            random.shuffle(events_origin)
            for event in events_origin[0:k]:
                train_events.append(event)
            for event in events_origin[k:k*2]:
                dev_events.append(event)
            for event in events_origin[k*2:event_num]:
                test_events.append(event)
        idx += event_num
    logger.info(f"Train={len(train_events)}, Dev={len(dev_events)}, Test={len(test_events)}")
    return train_events, dev_events, test_events

# test n*k-shot
def train_dev_nk_test_split(events, k: int):
    types_num = Counter([t.type for t in events]).most_common()
    train_events, dev_events, test_events = [], [], []
    idx = 0
    n = k // 100
    k = k % 100 # 104 means: n=1, k=4
    while idx < len(events):
        for item in types_num:
            if item[0] == events[idx].type:
                event_num = item[1]
                break
        if event_num >= k*(2+n):
            random.seed(RANDOM_STATE)
            events_origin = events[idx:idx+event_num]
            random.shuffle(events_origin)
            for event in events_origin[0:k]:
                train_events.append(event)
            for event in events_origin[k:k*2]:
                dev_events.append(event)
            for event in events_origin[k*2:k*(2+n)]:
                test_events.append(event)
        idx += event_num
    logger.info(f"Train={len(train_events)}, Dev={len(dev_events)}, Test={len(test_events)}")
    return train_events, dev_events, test_events


def main():
    for k in [4, 8, 16, 32]:
        for data_set in [
            DatasetEnum.Ace,
            DatasetEnum.FewShotED,
        ]:
            events = load_events(data_set, k)
            logger.info(f"Train={len(events.seen.train)}, Dev={len(events.seen.dev)}, Test={len(events.seen.test)}")
            print("end!!!")


if __name__ == '__main__':
    main()
