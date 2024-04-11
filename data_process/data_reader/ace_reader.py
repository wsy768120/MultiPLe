import json
from collections import Counter
from typing import List
import numpy as np
from tqdm import tqdm
from data_process.data_enum import DatasetEnum, SAMPLE_NUMBER_THRESHOLD
from data_process.event import Event, MAX_MENTION_LENGTH
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR
from tool.translator_helper import translator


def read_ace_data(file_path, es_translation, zh_translation) -> List[Event]:
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    events = []
    for e in tqdm(json_data, desc="gen_ace_events", ncols=100):
        if e["golden-event-mentions"]:
            mention = e['sentence']
            event_type = e["golden-event-mentions"][0]["event_type"]
            trigger = e["golden-event-mentions"][0]["trigger"]["text"]
            # translate en->zh,es
            zh_mention= zh_translation(mention, max_length=100)[0]['translation_text']
            zh_trigger = zh_translation(trigger, max_length=100)[0]['translation_text']
            es_mention = es_translation(mention, max_length=100)[0]['translation_text']
            es_trigger = es_translation(trigger, max_length=100)[0]['translation_text']

            event = Event("", event_type, mention, trigger, zh_mention, zh_trigger, es_mention, es_trigger)
            events.append(event)

    return events


def read_events(data_set: DatasetEnum) -> List[Event]:
    es_translation, zh_translation = translator()
    train_data = read_ace_data(ROOT_DIR.joinpath(f"data/{data_set.value}/train.json"), es_translation, zh_translation)
    dev_data = read_ace_data(ROOT_DIR.joinpath(f"data/{data_set.value}/dev.json"), es_translation, zh_translation)
    test_data = read_ace_data(ROOT_DIR.joinpath(f"data/{data_set.value}/test.json"), es_translation, zh_translation)
    all_data = train_data + dev_data + test_data
    type_count = Counter([e.type for e in all_data])
    all_data = [e for e in all_data if type_count[e.type] > SAMPLE_NUMBER_THRESHOLD]
    all_data = sorted(all_data, key=lambda e: e.type, reverse=True)

    for idx, e in enumerate(all_data):
        e.doc_id = f"Ace_{idx}"

    # analysis
    # type_count_sorted = Counter([e.type for e in all_data]).most_common()
    # type_count_list = [item[1] for item in type_count_sorted]
    # ins_max = type_count_list[0]
    # ins_mean = np.mean(type_count_list)
    # ins_std = np.std(type_count_list)
    # logger.info(f"ins_max={ins_max}, ins_mean={round(ins_mean, 2)}, ins_std={round(ins_std, 2)}")
    # trigger_length = [e.trigger_idx_range[1] - e.trigger_idx_range[0] + 1 for e in all_data]
    # total_trigger_length = np.sum(trigger_length)
    # max_trigger_length = np.max(trigger_length)
    # trigger_length_mean = round(total_trigger_length / len(trigger_length), 2)
    # logger.info(f"max_trigger_length={max_trigger_length}, trigger_length_mean={trigger_length_mean}")
    # logger.info(f"average_mention_length={round(total_mention_length / len(all_data), 2)}")
    # # draw
    # max_type = type_count_sorted[0][0]
    # max_type_events = [e for e in all_data if e.type == max_type]
    # trigger_of_max_type = Counter([e.trigger for e in max_type_events]).most_common()
    # print("trigger of '", max_type, "':\n", trigger_of_max_type)
    # trigger_count_sorted = Counter([e.trigger for e in all_data]).most_common()
    # max_trigger = trigger_count_sorted[0][0]
    # max_trigger_events = [e for e in all_data if e.trigger == max_trigger]
    # event_of_max_trigger = Counter([e.type for e in max_trigger_events]).most_common()
    # print("event of '", max_trigger, "':\n", event_of_max_trigger)
    return all_data


if __name__ == '__main__':
    logger.info(f"all_data={len(read_events(DatasetEnum.Ace))}")
