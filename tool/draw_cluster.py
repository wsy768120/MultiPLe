import pickle
import random
from typing import List
import matplotlib.pyplot as plt
import numpy
import torch
from collections import Counter
from matplotlib.axes import Axes
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_process.data_enum import DataTypeEnum
from data_process.data_reader.main_data_reader import load_events
from dataset.base_dataset import EventMentionEncoder
from tool.log_helper import logger
from tool.model_helper import load_model
from tool.path_helper import ROOT_DIR, mkdir_parent

RANDOM_STATE = 42
k = 150  # instances
q = 30  # labels

def k_test(events, k: int):
    types_num = Counter([t.type for t in events]).most_common()
    test_events = []
    idx = 0
    while idx < len(events):
        for item in types_num:
            if item[0] == events[idx].type:
                event_num = item[1]
                break
        # if event_num >= k:
        random.seed(RANDOM_STATE)
        events_origin = events[idx:idx+event_num]
        random.shuffle(events_origin)
        for event in events_origin[0:k]:
            test_events.append(event)
        idx += event_num
    return test_events

def predict(model_save_name: str, device: str = "cuda:0"):
    out_path = ROOT_DIR.joinpath(f"out/draw/vector_pt/{model_save_name}.pt")
    mkdir_parent(out_path)
    if out_path.exists():
        return

    model = load_model(model_save_name, device)
    logger.info(model_save_name)
    events = load_events(model.config.data_set, 4)
    test_events = events.seen.test
    sampled_events = k_test(test_events, k)
    encoder = EventMentionEncoder.get_encoder(model.config.plm)
    pin_memory = not model.config.use_cpu
    validator = model.get_validator(DataTypeEnum.Test)
    dataset = validator.dataset_class(sampled_events, encoder)
    dataloader = DataLoader(dataset, batch_size=model.config.eval_batch_size, pin_memory=pin_memory)

    model.eval()
    event_vectors = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model.move_batch_to_device(batch)
            # predict_trigger:
            soft_trigger = model.predict_trigger(batch['self_event'])
            predict_triggers = torch.argmax(soft_trigger, dim=-1)
            for i in range(batch['self_event']['input_ids'].shape[0]):
                batch['self_event']['input_ids'][i, 1] = predict_triggers[i]
            # predict_event:
            event_vector = model.predict_event_validator(batch['self_event'])
            event_vectors.append(event_vector)

    event_types = numpy.array([e.type for e in sampled_events])
    event_vectors = torch.cat(event_vectors).cpu().numpy()
    event_vectors = TSNE(n_components=2).fit_transform(event_vectors)
    pickle.dump((event_types, event_vectors), open(out_path, "wb"))


def draw_one(model_save_name: str, tag: str = ""):
    predict(model_save_name)
    input_path = ROOT_DIR.joinpath(f"out/draw/vector_pt/{model_save_name}.pt")
    types, vectors = pickle.load(open(input_path, "rb"))
    vectors_by_type = dict()
    pairs = list(zip(types, vectors))
    for t, v in pairs:
        if t not in vectors_by_type:
            vectors_by_type[t] = []
        vectors_by_type[t].append(v)

    vectors_by_type = dict(sorted(vectors_by_type.items(), key=lambda x: len(x[1]), reverse=True)[:q])
    # sample_types = random.Random(RANDOM_STATE).sample(vectors_by_type.keys(), q)
    plt.figure(figsize=(7, 7))
    for k, v in vectors_by_type.items():
        # if k in sample_types:
        x = [i[0] for i in v]
        y = [i[1] for i in v]
        plt.plot(x, y, "o", label=k)
    plt.axis('off')
    model_class = model_save_name.split("_")[0]
    # plt.title(model_class)
    plt.savefig(ROOT_DIR.joinpath(f"out/draw/cluster_{model_class}_{tag}.pdf"))


if __name__ == '__main__':
    draw_one("MultilingualModel_Ace_4shot_20230403104636721479", tag="ACE")
    draw_one("MultilingualModel_FewShotED_4shot_20230404100852598866", tag="FewShotED")
