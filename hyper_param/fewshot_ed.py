from model.base_model import BaseConfig

epoch_num = 100

multilingual_config = BaseConfig(
    learning_rate=1e-2,
    plm_learning_rate=1e-5,
    ec_weight=2.0
)