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

# Define where to save your models
save_path = '../../../../tudelft.net/staff-umbrella/MScThesisJLuu/models'


class HipXrayClassifier(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained ResNet18
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all kernels to finish
            allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
            reserved = torch.cuda.memory_reserved() / 1024 ** 2  # in MB
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2  # peak
            max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2

            print(
                f"[GPU Stats] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Allocated: {max_allocated:.2f} MB | Max Reserved: {max_reserved:.2f} MB")

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


def get_dataloaders(oai_files, check_files=None, batch_size=32):
    train_ds = HipXrayBinaryDataset(oai_files, split='train')
    val_ds = HipXrayBinaryDataset(oai_files, split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if check_files:
        check_val = HipXrayBinaryDataset(check_files, split='val')
        check_val_loader = DataLoader(check_val, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, check_val_loader

    return train_loader, val_loader, None


def train_model(oai_files, check_files=None, num_epochs=100, batch_size=32, lr=1e-4):
    train_loader, val_loader, check_val_loader = get_dataloaders(oai_files, check_files, batch_size)

    model = HipXrayClassifier(lr=lr)

    logger = TensorBoardLogger("logs", name="hip_xray_classifier")

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename='OA_classifier-{epoch:03d}',
        save_top_k=1,
        save_weights_only=True,
        verbose=True
    )

    early_stopping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=-1,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=5,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    return model
