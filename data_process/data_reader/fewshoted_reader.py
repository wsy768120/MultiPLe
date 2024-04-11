import json
from typing import List
from collections import Counter
import numpy as np
from tqdm import tqdm
from data_process.data_enum import DatasetEnum, SAMPLE_NUMBER_THRESHOLD
from data_process.event import Event, MAX_MENTION_LENGTH
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR
from tool.translator_helper import translator


def read_events(data_set: DatasetEnum) -> List[Event]:
    es_translation, zh_translation = translator()
    with open(ROOT_DIR.joinpath(f"data/{data_set.value}/Few-Shot_ED.json"), "r", encoding="utf-8") as f:
        json_data = json.load(f)

    events = []
    # total_mention_length = 0
    for event_type, event_mentions in tqdm(json_data.items(), desc="gen_fewshoted_events", ncols=100):
        if len(event_mentions) > SAMPLE_NUMBER_THRESHOLD:
            for idx, mention in tqdm(enumerate(event_mentions), desc=f"{event_type}", ncols=100):
                # total_mention_length += len(words)
                event_mention = mention[0].lower().split()
                event_mention = " ".join(event_mention[0:100])  # 防止超tranlator的max_encode_num:512

                # translate en->zh,es
                zh_mention = zh_translation(event_mention, max_length=100)[0]['translation_text']
                zh_trigger = zh_translation(mention[1], max_length=100)[0]['translation_text']
                es_mention = es_translation(event_mention, max_length=100)[0]['translation_text']
                es_trigger = es_translation(mention[1], max_length=100)[0]['translation_text']

                event = Event(f"FewShotED_{event_type}_{idx}", event_type, mention[0], mention[1], zh_mention, zh_trigger, es_mention, es_trigger)

                events.append(event)

    # analysis
    # type_count_sorted = Counter([e.type for e in events]).most_common()
    # type_count_list = [item[1] for item in type_count_sorted]
    # ins_max = type_count_list[0]
    # ins_mean = np.mean(type_count_list)
    # ins_std = np.std(type_count_list)
    # logger.info(f"ins_max={ins_max}, ins_mean={round(ins_mean, 2)}, ins_std={round(ins_std, 2)}")
    # trigger_length = [e.trigger_idx_range[1] - e.trigger_idx_range[0] + 1 for e in events]
    # total_trigger_length = np.sum(trigger_length)
    # max_trigger_length = np.max(trigger_length)
    # trigger_length_mean = round(total_trigger_length / len(trigger_length), 2)
    # logger.info(f"max_trigger_length={max_trigger_length}, trigger_length_mean={trigger_length_mean}")
    # logger.info(f"average_mention_length={round(total_mention_length / len(events), 2)}")

    return events


if __name__ == '__main__':
    logger.info(f"all_data={len(read_events(DatasetEnum.FewShotED))}")
