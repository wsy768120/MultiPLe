from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from data_process.data_enum import DatasetEnum, DataTypeEnum, PlmEnum
from tool.log_helper import logger
from tool.pretrain_model_helper import load_model


class DistFuncEnum(Enum):
    CosineSimilarity = "CosineSimilarity"
    Euclidean = "Euclidean"
    KullbackLeibler = "Kullback–Leibler"
    Wasserstein = "Wasserstein"


def wasserstein_distance(x: torch.Tensor, y: torch.Tensor, p: int = 1):
    # From https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss
    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    x = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-14)
    y = y / (torch.sum(y, dim=-1, keepdim=True) + 1e-14)

    # make cdf with cumsum
    cdf_p1 = torch.cumsum(x, dim=-1)
    cdf_p2 = torch.cumsum(y, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_p1 - cdf_p2)), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_p1 - cdf_p2), 2), dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_p1 - cdf_p2), p), dim=-1), 1 / p)

    return cdf_distance


def kl_divergence(x: torch.Tensor, y: torch.Tensor):
    kl_div = (x * torch.log(x / y)).sum(dim=1)
    return kl_div


DistFuncMap = {
    DistFuncEnum.CosineSimilarity: torch.cosine_similarity,
    DistFuncEnum.Euclidean: torch.pairwise_distance,
    DistFuncEnum.KullbackLeibler: kl_divergence,
    DistFuncEnum.Wasserstein: wasserstein_distance,
}


@dataclass
class BaseConfig(object):
    data_set: DatasetEnum = DatasetEnum.Ace
    plm: PlmEnum = PlmEnum.Bert
    plm_fix_layers_ratio: float = 2 / 3
    epoch_num: int = 100
    check_point_step: int = 500
    early_stop_check_point: int = 10
    train_batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    plm_learning_rate: float = 1e-6
    weight_decay: float = 0
    main_device: str = "cpu"
    multi_device_ids: List[int] = None
    random_seed: int = None
    unseen_type_num: int = 0
    seen_type_num: int = 0
    parent_type_num: int = 0
    parent_child_list: list = None
    ec_weight: float = 1.0
    contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean
    k: int = 0


    @property
    def use_cpu(self):
        return self.main_device in ["cpu", "CPU"]

    @property
    def k_num(self) -> int:
        k_num = self.seen_type_num + self.unseen_type_num
        return k_num


class BaseMeta(type):
    """
    Let str(BaseModel) returns "BaseModel".
    See: https://stackoverflow.com/a/63195609/6907563
    """

    def __str__(self):
        return self.__name__


class BaseModel(torch.nn.Module, metaclass=BaseMeta):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.current_training_step = 0
        self.config = config
        self.plm_for_mask = load_model(config.plm)
        if self.config.plm.value in ["bert-base-uncased", "bert-base-multilingual-cased"] :
            self.plm_model = self.plm_for_mask.bert
            self.plm_cls = self.plm_for_mask.cls
        else:
            self.plm_model = self.plm_for_mask.roberta
            self.plm_cls = self.plm_for_mask.lm_head
        self.fix_plm_param(self.plm_model)

    def get_name(self):
        return f"{self.__class__.__name__}_{self.config.data_set}_{self.config.k}shot"

    def get_device(self) -> torch.device:
        return list(self.parameters())[0].device

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def move_batch_to_device(self, batch):
        for k, v in batch.items():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = v.to(self.config.main_device)
            else:  # Dict[str, Tensor]
                self.move_batch_to_device(batch[k])

    def fix_plm_param(self, plm_model):
        if self.config.plm_fix_layers_ratio > 0:
            plm_layers_n = len(plm_model.encoder.layer)
            fix_layers_n = int(self.config.plm_fix_layers_ratio * plm_layers_n)
            layers_require_grad = [str(idx) for idx in range(fix_layers_n, plm_layers_n)] + ["pooler"]
            for name, param in plm_model.named_parameters():
                param.requires_grad = False
                for layer in layers_require_grad:
                    layer_name = set(name.split("."))
                    if layer in layer_name:
                        param.requires_grad = True

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        raise NotImplementedError()

    def get_validator(self, datatype: DataTypeEnum):
        raise NotImplementedError()

    def encode(self, event_encoding: Dict[str, torch.Tensor]):
        plm_out = self.plm_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"],
            event_encoding["token_type_ids"])

        mask = event_encoding["trigger_mask"].unsqueeze(2)
        event_vec = plm_out.last_hidden_state * mask
        event_vec = event_vec.sum(dim=1) / mask.sum(dim=1)

        # event_vec shape check
        batch_size = event_encoding["input_ids"].shape[0]
        plm_hidden_dim = plm_out.last_hidden_state.shape[2]
        assert event_vec.shape == (batch_size, plm_hidden_dim)

        return event_vec

    def classify(self, event_encoding: Dict[str, torch.Tensor]):
        # 输出样本分类的概率
        raise NotImplementedError()


def unit_test(model: BaseModel):
    model.to(model.config.main_device)

    # Train
    model.train()
    dataset = model.create_dataset(DataTypeEnum.Train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch = next(iter(dataloader))
    model.move_batch_to_device(batch)
    loss = model(**batch)
    assert loss.shape == (64,)
    loss = loss.mean()
    loss.backward()
    logger.info(loss)

    # Eval
    model.eval()
    validator = model.get_validator(DataTypeEnum.Dev)
    logger.info(validator.eval())
