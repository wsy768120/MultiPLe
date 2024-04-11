from typing import Type
import numpy
import torch
from data_process.data_enum import DataTypeEnum
from dataset.base_validator_dataset import BaseValidatorDataset
from tool.log_helper import logger
from tool.model_helper import load_model
from validator.base_validator import BaseValidator


class WoHierarchicalValidator(BaseValidator):
    def __init__(self, model, datatype: DataTypeEnum, dataset_class: Type[BaseValidatorDataset] = BaseValidatorDataset):
        super().__init__(model, datatype, dataset_class)

    def encode_event_with_dataloader(self, events, dataloader):
        self.model.eval()
        predict_labels = []
        with torch.no_grad():
            for batch in dataloader:
                self.model.move_batch_to_device(batch)
                # predict_trigger:
                soft_trigger = self.model.predict_trigger(batch['self_event'])
                predict_triggers = torch.argmax(soft_trigger, dim=-1)
                for i in range(batch['self_event']['input_ids'].shape[0]):
                    batch['self_event']['input_ids'][i, 1] = predict_triggers[i]

                # predict_event:
                event_emb = self.model.predict_event_validator(batch['self_event'])
                soft_label = self.model.cal_cluster_prob(event_emb)
                predict_labels.append(torch.argmax(soft_label, dim=-1))

        actual_labels = [event.type_id for event in events]
        predict_labels = torch.cat(predict_labels).cpu()

        return numpy.array(actual_labels), predict_labels.numpy()


def main(model_save_name: str, device: str = "cuda:0"):
    model = load_model(model_save_name, device)
    logger.info(model_save_name)
    logger.info(model.config)
    validator = WoHierarchicalValidator(model, DataTypeEnum.Test)
    logger.info(validator.eval())


if __name__ == '__main__':
    for model_name in []:
        main(model_name)
