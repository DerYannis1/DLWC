import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleWeatherCNN(pl.LightningModule):
    def __init__(self, in_channels=35, out_channels=35):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(in_channels, 200, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200, 400, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(400)
        self.conv3 = nn.Conv2d(400, 400, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(400)
        self.conv4 = nn.Conv2d(400, 200, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(200)
        self.conv5 = nn.Conv2d(200, 100, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(100)
        self.conv6 = nn.Conv2d(100, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x

    def training_step(self, batch, batch_idx):
        inp, out, *_ = batch
        pred = self(inp)
        loss = F.mse_loss(pred, out)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, out, *_ = batch
        pred = self(inp)
        loss = F.mse_loss(pred, out)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
