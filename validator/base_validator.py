from typing import Type
import torch
import numpy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import *
from torch.utils.data import DataLoader
import random
import json
from data_process.data_enum import DataTypeEnum
from data_process.data_reader.main_data_reader import load_event_list
from dataset.base_dataset import EventMentionEncoder
from dataset.base_validator_dataset import BaseValidatorDataset
from tool.log_helper import logger
from collections import Counter


class BaseValidator:
    def __init__(self, model, datatype: DataTypeEnum, dataset_class: Type[BaseValidatorDataset]):
        self.model = model
        self.dataset_class = dataset_class
        self.encoder = EventMentionEncoder.get_encoder(self.model.config.plm)
        self.events = load_event_list(self.model.config.data_set, datatype, self.model.config.k)

    def encode_event(self, events):
        config = self.model.config
        pin_memory = not config.use_cpu
        dataset = self.dataset_class(events, self.encoder)
        dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, pin_memory=pin_memory)
        return self.encode_event_with_dataloader(events, dataloader)

    def encode_event_with_dataloader(self, events, dataloader):
        self.model.eval()
        predict_labels = []
        with torch.no_grad():
            for batch in dataloader:
                self.model.move_batch_to_device(batch)
                soft = self.model.classify(batch)
                predict_labels.append(torch.argmax(soft, dim=-1))

        actual_labels = [e.type_id for e in events]
        predict_labels = torch.cat(predict_labels).cpu()
        return numpy.array(actual_labels), predict_labels.numpy()

    def eval(self):
        metrics = self.eval_seen()
        metrics.update(metrics)
        return metrics

    def eval_seen(self):
        seen_actual, seen_predict = self.encode_event(self.events)
        # # change_input_length：
        # events = self.change_input_length(self.events)
        # seen_actual, seen_predict = self.encode_event(events)
        # # IUS/TUS/COS:
        # events = self.challenge_sampling(self.events, 'COS')
        # seen_actual, seen_predict = self.encode_event(events)
        class_metric = calculate_classification_metric(seen_predict, seen_actual)
        return class_metric

    def encode_event(self, events):
        config = self.model.config
        pin_memory = not config.use_cpu
        dataset = self.dataset_class(events, self.encoder)
        dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, pin_memory=pin_memory)
        return self.encode_event_with_dataloader(events, dataloader)

    def change_input_length(self, events):
        x = 50  # 10,20,30,40,50,60
        # test_events = [e for e in events if (len(e.mention.split()) <= x and len(e.mention.split()) > x - 10)]
        test_events = [e for e in events if len(e.mention.split()) > x]
        logger.info(f"test_events: {len(test_events)}\n")
        return test_events

    # IUS/TUS/COS:
    def challenge_sampling(self, events, method: str):
        # IUS: from each event type sample k instances.
        # TUS: from each event type sample k triggers regardless of their occurring frequency. one trigger one instance.
        # COS: from each event type sample k confusing triggers. one trigger one instance.
        k = 4
        test_events = []
        types_num = Counter([e.type for e in events])
        idx = 0
        for value, num in types_num.items():
            if (events[idx].type == value) and (num >= k):
                if method == "IUS":
                    random.seed(2020)
                    ius_events = events[idx: idx + num]
                    random.shuffle(ius_events)
                    for event in ius_events[:k]:
                        test_events.append(event)
                else:
                    trigger_num = Counter([e.trigger for e in events[idx:idx+num]])
                    trigger_list = [item for item in trigger_num]
                    random.seed(2020)
                    if method == "TUS":
                        random.shuffle(trigger_list)
                    if method == "COS":
                        other_triggers_num = Counter([e.trigger for e in events if e.type != value])
                        other_triggers_list = [item for item in other_triggers_num]
                        trigger_list = self.confusing_trigger(trigger_list, other_triggers_list)
                    k_trigger = trigger_list[0:k]
                    for trigger in k_trigger:
                        event = [e for e in events if e.trigger == trigger]
                        random.shuffle(event)
                        test_events.append(event[0])
            idx += num
        logger.info(f"test_events: {len(test_events)}\n")
        return test_events

    def confusing_trigger(self, trigger_list, other_triggers_list):
        word2id = json.load(open('./tool/word2id.json'))
        word_emb = numpy.load('./tool/word_embedding.npy')
        k = 20

        triggers1 = list(filter(lambda x: x in word2id, trigger_list))
        t1_embs = torch.tensor([word_emb[word2id[t]] for t in triggers1])
        inner_dist = torch.norm(t1_embs.unsqueeze(1) - t1_embs.unsqueeze(0).expand(t1_embs.size(0), -1, -1), dim=-1, p=2)  # k1 x k1
        k1, _ = inner_dist.size()
        topk_inner_dist = torch.sort(inner_dist, dim=-1)[0][:, :min(k1, k)]
        inner_dist_mean = torch.mean(topk_inner_dist, dim=-1)

        triggers2 = list(filter(lambda x: x in word2id, other_triggers_list))
        t2_embs = torch.tensor([word_emb[word2id[t]] for t in triggers2])
        inter_dist = torch.norm(t1_embs.unsqueeze(1) - t2_embs.unsqueeze(0).expand(t1_embs.size(0), -1, -1), dim=-1, p=2)  # k1 x k2
        k2, _ = inter_dist.size()
        topk_inter_dist = torch.sort(inter_dist, dim=-1)[0][:, :min(k2, k)]
        inter_dist_mean = torch.mean(topk_inter_dist, dim=-1)

        dist = -inner_dist_mean + inter_dist_mean
        chose_index = torch.topk(dist, k=len(dist), largest=False)[1]
        confusing_trigger_list = [triggers1[i] for i in chose_index]
        return confusing_trigger_list


def calculate_classification_metric(actual_labels, predict_labels, tag: str = "") -> dict:
    result = dict()
    p, r, f, _ = precision_recall_fscore_support(actual_labels, predict_labels, average='weighted', zero_division=0)
    result["acc" + tag] = accuracy_score(actual_labels, predict_labels)
    result["pre" + tag] = precision_score(actual_labels, predict_labels, average='weighted', zero_division=0)
    result["rec" + tag] = recall_score(actual_labels, predict_labels, average='weighted', zero_division=0)
    result["f1" + tag] = f1_score(actual_labels, predict_labels, average='weighted', zero_division=0)
    return result


def calculate_cluster_metric(actual_labels: numpy.ndarray, predict_labels: numpy.ndarray) -> dict:
    cluster_metrics = {
        rand_score: "rand",
        adjusted_rand_score: "arand",
        normalized_mutual_info_score: "nmi",
        adjusted_mutual_info_score: "anmi",
        fowlkes_mallows_score: "fm",
        completeness_score: "comp",
        homogeneity_score: "homo",
        v_measure_score: "vm"
    }

    result = {name: metric(actual_labels, predict_labels) for metric, name in cluster_metrics.items()}

    nClass = max(actual_labels.max(), predict_labels.max()) + 1
    weight_M = numpy.zeros((nClass, nClass))
    for i in range(len(actual_labels)):
        weight_M[predict_labels[i], actual_labels[i],] += 1
    ind = linear_sum_assignment(weight_M.max() - weight_M)
    best_map = {ind[0][i]: ind[1][i] for i in range(nClass)}
    best_map_labels = [best_map[x] for x in predict_labels]
    result.update(calculate_classification_metric(actual_labels, best_map_labels, "_cluster"))

    return result


def main():
    metric = calculate_cluster_metric(numpy.array([1, 2, 3, 4, 5]), numpy.array([5, 1, 2, 3, 4]))
    logger.info(metric)
    assert metric["f1"] == 1


if __name__ == '__main__':
    main()
