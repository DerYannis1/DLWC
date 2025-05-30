import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleWeatherCNN(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inp, out, *_ = batch  # Nur inp und out verwenden
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