import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, roc_auc_score
from data_handler import HipXrayBinaryDataset
from pytorch_lightning.loggers import TensorBoardLogger
import os

# Define where to save your models
save_path = '../../../../tudelft.net/staff-umbrella/MScThesisJLuu/models'


class HipXrayClassifier(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load ResNet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)  # make sure target shape matches output
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.sigmoid(logits)
        acc = accuracy_score(y.cpu(), (preds.cpu() > 0.5).int())

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = y.float().unsqueeze(1)
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.sigmoid(logits)
        acc = accuracy_score(y.cpu(), (preds.cpu() > 0.5).int())
        auc = roc_auc_score(y.cpu(), preds.cpu())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)


def get_dataloaders(files, batch_size=32):
    train_ds = HipXrayBinaryDataset(files, split='train')
    val_ds = HipXrayBinaryDataset(files, split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, None


def train_model(oai_files, num_epochs=100, batch_size=32, lr=1e-4):
    train_loader, val_loader, check_val_loader = get_dataloaders(oai_files, batch_size)

    model = HipXrayClassifier(lr=lr)

    logger = TensorBoardLogger("logs", name="hip_xray_classifier")

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath='',
        filename='OA_classifier-{epoch:03d}',
        save_top_k=1,
        save_weights_only=True,
        verbose=True
    )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    return model
