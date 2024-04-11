import random
import numpy
import torch
from data_process.data_enum import DatasetEnum, PlmEnum
from data_process.data_reader.main_data_reader import load_events
from hyper_param import Ace, fewshot_ed
from model.base_model import BaseConfig, BaseModel
from model.multilingual import MultilingualModel
from model.ablation.wocontrastive import WoContrastiveModel
from model.ablation.wohierarchical import WoHierarchicalModel
from model.ablation.womultilingual import WoMultilingualModel
from model.ablation.wochinese import WoChineseModel
from model.ablation.wospanish import WoSpanishModel
from tool.train_helper import TrainHelper

TRAIN_RANDOM_SEED = 2020

hyper_param_map = {
    DatasetEnum.Ace: Ace,
    DatasetEnum.FewShotED: fewshot_ed,
}


def train_with_config(model_class: type, config: BaseConfig, data_set: DatasetEnum, plm_class: PlmEnum, k: int, train_tag: str = None):
    random.seed(TRAIN_RANDOM_SEED)
    numpy.random.seed(TRAIN_RANDOM_SEED)
    torch.manual_seed(TRAIN_RANDOM_SEED)
    torch.cuda.manual_seed(TRAIN_RANDOM_SEED)

    update_common_config(config, data_set, plm_class, k)
    model: BaseModel = model_class(config)
    model.to(config.main_device)
    train_helper = TrainHelper(model, train_tag)
    train_helper.train_model()


def update_common_config(config: BaseConfig, data_set: DatasetEnum, plm_class: PlmEnum, k: int):
    config.epoch_num = hyper_param_map[data_set].epoch_num
    config.train_batch_size = 4
    config.main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.multi_device_ids = [0, 1, 2, 3]
    config.eval_batch_size = config.train_batch_size * 4
    config.random_seed = TRAIN_RANDOM_SEED
    config.check_point_step = 100
    config.early_stop_check_point = 10
    config.data_set = data_set
    config.weight_decay = 1e-6
    config.plm = plm_class
    config.plm_fix_layers_ratio = 0
    config.k = k
    all_events = load_events(data_set, k)
    type_id_set = set([(e.type, e.type_id) for e in all_events.seen.train])
    config.seen_type_num = len(type_id_set)
    config.parent_child_list, config.parent_type_num = parent_type(type_id_set, data_set)


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


def main_train(model_class: type, data_set: DatasetEnum, plm_class: PlmEnum, k: int, train_tag: str = None):
    hp = hyper_param_map[data_set]

    default_config_map = {
        MultilingualModel: hp.multilingual_config,
        WoContrastiveModel: hp.multilingual_config,
        WoMultilingualModel: hp.multilingual_config,
        WoHierarchicalModel: hp.multilingual_config,
        WoChineseModel: hp.multilingual_config,
        WoSpanishModel: hp.multilingual_config,
    }

    config = default_config_map[model_class]
    train_with_config(model_class, config, data_set, plm_class, k, train_tag)


if __name__ == '__main__':
    # # ablation:
    # main_train(WoContrastiveModel, DatasetEnum.FewShotED, PlmEnum.XLM, 4)
    # main_train(WoHierarchicalModel, DatasetEnum.FewShotED, PlmEnum.XLM, 4)
    # main_train(WoMultilingualModel, DatasetEnum.FewShotED, PlmEnum.XLM, 4)
    # main_train(WoChineseModel, DatasetEnum.FewShotED, PlmEnum.XLM, 4)
    # main_train(WoSpanishModel, DatasetEnum.FewShotED, PlmEnum.XLM, 4)

    # # overall:
    for k in [4, 8, 16, 32]:
        main_train(MultilingualModel, DatasetEnum.FewShotED, PlmEnum.XLM, k)
        main_train(MultilingualModel, DatasetEnum.Ace, PlmEnum.XLM, k)







