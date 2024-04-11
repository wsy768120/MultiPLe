import torch
from typing import Dict
from torch.utils.data import Dataset
from data_process.data_enum import DataTypeEnum
from dataset.multilingual_dataset import MultilingualDataset
from dataset.base_validator_dataset import BaseValidatorDataset
from model.base_model import BaseModel, BaseConfig, DistFuncEnum, DistFuncMap, unit_test
from validator.multilingual_validator import MultilingualValidator

# multilingual prompt -> ee prompt
# ordered contrastive learning -> ml to ee
# hierarchical prototypical network

class WoChineseModel(BaseModel):
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.config = config
        self.trigger_mask_idx = 1
        self.event_mask_idx = 4
        self.mention_offset = 8
        self.p_dist_func = DistFuncMap[DistFuncEnum.Euclidean]
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.nll_loss = torch.nn.NLLLoss(reduction="none")
        self.child_center = torch.nn.Parameter(torch.randn([config.k_num, int(self.plm_model.config.hidden_size)]))  # (32,1024)
        self.parent_center = torch.nn.Parameter(torch.randn([config.parent_type_num, int(self.plm_model.config.hidden_size)]))  # (8,1024)
        self.contrastive_dist_func = DistFuncEnum.Euclidean
        self.contrastive_loss_margin: float = 2.0
        self.ep_weight = 1.0
        self.tl_weight = 1.0
        self.el_weight = 1.0
        self.cl_weight = 2.0
        self.device = torch.device('cuda:0')

    def forward(self, self_event, self_trigger, zh_event, es_event, neu_event, neg_event, self_trigger_label, self_event_label):
        # trigger identification:
        predict_trigger = self.predict_trigger(self_trigger)
        trigger_loss = self.ce_loss(predict_trigger, self_trigger_label)

        # event classification:
        emb_en, emb_ee = self.get_event_embedding(self_event, es_event)
        p_parent, p_child = self.cal_cluster_prob(emb_en)
        event_parent_label = self.parent_label(self_event_label)
        event_parent_loss = self.nll_loss(torch.log(p_parent), event_parent_label)
        event_child_loss = self.nll_loss(torch.log(p_child), self_event_label.squeeze(1))
        event_loss = self.ep_weight * event_parent_loss + self.config.ec_weight * event_child_loss

        # 计算有序对比损失contrastive_loss
        contrastive_loss = self.quadruple_loss(p_child, emb_ee, neu_event, neg_event)

        return self.tl_weight * trigger_loss + self.el_weight * event_loss + self.cl_weight * contrastive_loss

    def predict_trigger(self, event_encoding: Dict[str, torch.Tensor]):
        xlm_out = self.plm_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"]
        )
        prediction_scores = self.plm_cls(xlm_out[0])  # xlm_out[0](4,50,1024) to prediction_scores(16,50,250002)
        logit = prediction_scores[:, self.trigger_mask_idx, :]  # 第0维取所有:，第1维取1,代表trigger的<MASK>，第2维取所有:. logit: Tensor(4,250002)
        mention_ids = self.mention_loc(event_encoding["input_ids"])  # tensor(4,50)
        #只有mention中的vocab对应的token为1, 其他元素为0. scatter(dim, index, src)将src中数据根据index中的索引按照dim的方向进行填充。 mention_mask: Tensor(4,250002)
        mention_mask = torch.zeros_like(logit).scatter(1, mention_ids, 1)
        mention_mask[:, 0] = 0  # set [PAD] mask to False
        logit *= mention_mask # [mask]只能预测input中的vocab
        return logit  # Tensor(4,250002)

    def mention_loc(self, embedding: torch.Tensor):
        mention_ids_list = []
        for i in range(embedding.shape[0]):  # 16
            row = embedding[i, :].tolist()
            for j in range(self.mention_offset):  # [cls] <mask> triggers <mask> event. </s> ... # 0, 250001,185553,7,250001,19732,5,2
                row[j] = row[self.mention_offset]
            mention_id = torch.tensor([row], dtype=torch.long).to(self.device)
            mention_ids_list.append(mention_id)
        mention_ids = torch.cat(mention_ids_list, dim=0).to(self.device)
        return mention_ids

    def get_event_embedding(self, self_event, es_event):
        emb_en = self.predict_event(self_event)
        emb_es = self.predict_event(es_event)
        # 拼接多语种事件表征，得到联合事件表征
        emb_ee = torch.divide(torch.add(emb_en, emb_es), 2)
        return emb_en, emb_ee

    def predict_event(self, event_encoding: Dict[str, torch.Tensor]):
        xlm_out = self.plm_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"]
        )
        mask_vector_list = []
        for i in range(xlm_out.last_hidden_state.shape[0]):  # xlm_out.last_hidden_state (4,50,1024)
            mask_idx = event_encoding["input_ids"][i].tolist().index(250001)
            mask_vector = xlm_out.last_hidden_state[i, mask_idx, :]
            mask_vector_list.append(mask_vector.unsqueeze(0))
        event_vector = torch.cat(mask_vector_list, dim=0).to(self.device)
        return event_vector  # tensor(4,1024)

    def predict_event_validator(self, event_encoding: Dict[str, torch.Tensor]):
        xlm_out = self.plm_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"]
        )
        event_vector = xlm_out.last_hidden_state[:, self.event_mask_idx, :]
        return event_vector  # tensor(16,1024)

    def cal_cluster_prob(self, event_embedding):  # event_embedding:(4,1024)
        # Hierarchical Prototypical Network
        # Parent_level:1024
        parent_hidden = event_embedding.unsqueeze(1).repeat(1, self.config.parent_type_num, 1)
        parent_center_batch = self.parent_center.unsqueeze(0).repeat(parent_hidden.shape[0], 1, 1)
        parent_dist = self.p_dist_func(parent_hidden, parent_center_batch)
        parent_probability = torch.softmax(-parent_dist, dim=1)

        # Child_level:1024
        child_hidden = event_embedding.unsqueeze(1).repeat(1, self.config.k_num, 1)
        child_center_batch = self.child_center.unsqueeze(0).repeat(child_hidden.shape[0], 1, 1)
        child_dist = self.p_dist_func(child_hidden, child_center_batch)
        child_probability = torch.softmax(-child_dist, dim=1)

        return parent_probability, child_probability

    def parent_label(self, child_label: torch.Tensor):

        parent_label_list = []
        list = []
        for i in range(self.config.parent_type_num):
            for j in self.config.parent_child_list[i]:
                list.append([j, i])
        for k in range(child_label.shape[0]):
            for item in list:
                if item[0] == child_label[k].item():
                    parent_label = item[1]
            parent_label_list.append(parent_label)
        event_parent_label = torch.tensor(parent_label_list).to(self.device)
        return event_parent_label

    def quadruple_loss(self, p_en, emb_ee, neu_event, neg_event):
        emb_neu = self.predict_event(neu_event)
        emb_neg = self.predict_event(neg_event)

        _, p_ee = self.cal_cluster_prob(emb_ee)
        _, p_neu = self.cal_cluster_prob(emb_neu)
        _, p_neg = self.cal_cluster_prob(emb_neg)

        dist_func = DistFuncMap[self.contrastive_dist_func]
        dist_ee = dist_func(p_en, p_ee)

        dist_neu = dist_func(p_en, p_neu)
        diff_neu = self.contrastive_loss_margin - (dist_neu - dist_ee)
        loss_neu = torch.max(torch.zeros_like(diff_neu), diff_neu)

        dist_neg = dist_func(p_en, p_neg)
        diff_neg = self.contrastive_loss_margin - (dist_neg - dist_neu)
        loss_neg = torch.max(torch.zeros_like(diff_neg), diff_neg)

        return dist_ee + loss_neu + loss_neg

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return MultilingualDataset(self.config.data_set, data_type, self.config.plm, self.config.k)

    def get_validator(self, datatype: DataTypeEnum):
        return MultilingualValidator(self, datatype, BaseValidatorDataset)


def main():
    config = BaseConfig()
    model = WoChineseModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
