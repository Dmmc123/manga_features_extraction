import torch.optim as optim
from torchvision import transforms, models
import numpy as np
import torchvision
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from PIL import Image
from random import choice
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss

device = "cuda:0"

# Following the tutorial from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

class EmbeddingModel(pl.LightningModule):
    def __init__(self):
        # model based on the model from paper https://paperswithcode.com/paper/making-classification-competitive-for-deep
        super().__init__()
        self.loss = NormalizedSoftmaxLoss(num_classes=103, embedding_size=2048).to(device)
        self.model = models.resnet50(pretrained=True)

        # # changing the first layer to accept 1 channel images
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.norm = nn.LayerNorm(1000)
        self.out = nn.Linear(1000, 2048, bias=False)
        
    def forward(self, x):
        x = self.model(x)
        x = self.norm(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        emb = self(x)
        loss = self.loss(emb, y)
        torch.cuda.empty_cache()
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        emb = self(x)
        loss = self.loss(emb, y)
        torch.cuda.empty_cache()
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        emb = self(x)
        loss = self.loss(emb, y)
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        # SGD for some reason appeared to be way more effective for the dataset
        return torch.optim.SGD(list(self.parameters()) + list(self.loss.parameters()), lr=0.01, weight_decay=.0001, momentum=0.9)
        # Adam, on the other hand, was seen to be useless compared with SGD
        # return torch.optim.Adam(list(self.parameters()) + list(self.loss.parameters()), lr=0.01)
        
class ImagesDataset(Dataset):
    """Manga dataset"""
    
    def __init__(self, path='data'):
        self.dataset = list(map(str, Path(path).rglob("*.jpg"))) +  list(map(str, Path(path).rglob("*.png")))
        items = set(map(lambda x: x.split(os.sep)[-2], self.dataset))
        self.name_to_label = dict(zip(items, range(len(items))))
        self.transform = transforms.Compose([
                    transforms.Resize((200, 200)),
                    transforms.ToTensor(),
                ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.dataset[idx]
        class_ = self.name_to_label[img_path.split(os.sep)[-2]]
        with Image.open(img_path) as img:

            rgb_img = Image.new("RGB", img.size)
            rgb_img.paste(img)
            img = rgb_img

            return (self.transform(img), class_)
        
# Following the tutorial from https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
class ImageRetreivalDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = 0.8
        self.test_size = 0.1

    def setup(self, stage=None):
        data_full = ImagesDataset()
        # Cropping for debug purpoces
        # data_full, _ = random_split(data_full, [int(len(data_full)*0.1), len(data_full) - int(len(data_full)*0.1)])
        train_size = int(len(data_full)*self.train_size)
        test_size = int(len(data_full)*self.test_size)
        val_size = len(data_full) - train_size - test_size
        self.image_retr_train, test_val = random_split(data_full, [train_size, test_size + val_size])
        self.image_retr_test, self.image_retr_val = random_split(test_val, [test_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.image_retr_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.image_retr_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.image_retr_test, batch_size=self.batch_size, num_workers=4)
