from config import Config
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Model
from data_load import CustomDataset
from orpo import ORPO


class TrainORPO:
    def __init__(self):
        self.config = Config()
        # 演员和评论家模型
        self.model = Model(self.config)
        self.tokenizer = self.model.tokenizer
        # 获得策略模型优化器, 这里使用的是lora, 不优化全量数据
        self.model_opt = Adam(self.model.parameters(), lr=self.config.lr)
        # 训练数据
        dataset = CustomDataset(self.config.data_path, self.tokenizer)
        self.data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True,
                                      collate_fn=dataset.collate_fn)
        self.orpo = ORPO(self.model, self.model_opt, self.config, self.tokenizer)

    def train_orpo(self):
        for epoch in range(self.config.epochs):
            for batch_data in self.data_loader:
                self.orpo.train(batch_data["inputs_ids"], batch_data["inputs_masks"], batch_data["labels_mask"])

        self.save_model()

    def save_model(self):
        # 保存lora参数
        self.model.model.save_pretrained(self.config.save_lora_path, safe_serialization=False)


if __name__ == '__main__':
    train_orpo = TrainORPO()
    train_orpo.train_orpo()
