import json

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import roc_curve
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from torchvision.models import resnet18, ResNet18_Weights

import torchvision.transforms as T

from train_models.data_utils.data_module import HipXrayDataModule

# Define where to save your models
save_path = '../../../../../tudelft.net/staff-umbrella/MScThesisJLuu/models'


class HipXrayClassifier(L.LightningModule):
    def __init__(self, epochs=100, lr=5e-5, filename=None, weight_decay=1e-1, dropout_prob=0.5):
        super().__init__()
        self.tested_source = False
        self.filename = filename
        self.test_outputs = []
        self.save_hyperparameters()

        # Load ResNet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 1 label per sample
        self.model.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        # metrics
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_loss_metric = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        y = y.float().unsqueeze(1)

        loss = self.bce_loss(logits, y)

        self.log("train_loss", loss, prog_bar=False)

        # Calculate predictions
        y_probs = torch.sigmoid(logits)
        y_preds = (y_probs > 0.5).float()

        # Update metrics
        self.train_acc.update(y_preds, y)
        self.train_auc.update(y_probs, y)

        return loss

    def on_train_epoch_end(self):
        # Calculate and log accuracy
        train_acc = self.train_acc.compute()
        train_auc = self.train_auc.compute()

        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_auc", train_auc, prog_bar=False)

        # Reset metrics
        self.train_acc.reset()
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        y = y.float().unsqueeze(1)

        # Calculate loss
        loss = self.bce_loss(logits, y)

        self.val_auc.update(probs, y)
        self.val_acc.update((probs > 0.5).float(), y)
        self.val_loss_metric.update(loss)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_auc = self.val_auc.compute()
        self.log("val_acc", val_acc, prog_bar=False)
        self.log("val_auc", val_auc, prog_bar=True)
        self.val_acc.reset()
        self.val_auc.reset()

        avg_loss = self.val_loss_metric.compute()
        self.log("val_loss", avg_loss, prog_bar=True)
        self.val_loss_metric.reset()

    def on_test_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        y = y.float().unsqueeze(1)
        self.test_outputs.append((probs.cpu(), y.cpu()))
        self.test_auc.update(probs, y)
        self.test_acc.update((probs > 0.5).float(), y)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_auc = self.test_auc.compute()

        self.log("test_acc", test_acc, prog_bar=True)
        self.log("test_auc", test_auc, prog_bar=True)

        self.test_acc.reset()
        self.test_auc.reset()

        # create ROC graph
        probs, labels = zip(*self.test_outputs)
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()

        fpr, tpr, thresholds = roc_curve(labels, probs)
        # Save ROC values for later comparison
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(test_auc),
        }

        if not self.tested_source:
            with open(f"train_models/results/graphs/source_{self.filename}_roc_values.json", "w") as f:
                json.dump(roc_data, f)
            self.tested_source = True
        else:
            with open(f"train_models/results/graphs/target_{self.filename}_roc_values.json", "w") as f:
                json.dump(roc_data, f)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[15], gamma=0.5
        # )
        # return [optimizer], [scheduler]
        return optimizer


def get_dataloaders(source, target, batch_size=32, source_ratio=0.5, source_majority_class=0, target_majority_class=0, transform=None):
    source_datamodule = HipXrayDataModule(source, train_split='train1', test_split='test1', val_split='val1', source_majority_class=source_majority_class, target_majority_class=target_majority_class, transform=transform)
    target_datamodule = HipXrayDataModule(source, train_split='train2', test_split='test2', val_split='val2', source_majority_class=source_majority_class, target_majority_class=target_majority_class)

    source_datamodule.setup()
    target_datamodule.setup()

    source_train_loader = source_datamodule.train_dataloader()

    source_val_loader = source_datamodule.val_dataloader()
    # target_val_loader = target_datamodule.val_dataloader()

    source_test_loader = source_datamodule.test_dataloader()
    target_test_loader = target_datamodule.test_dataloader()


    return source_train_loader, source_val_loader, source_test_loader, target_test_loader


def train_model(source_files, target_files, filename, num_epochs=100, batch_size=32, learning_rate=1e-4,
                source_ratio=0.5, majority_class=0, source_majority_class=0, target_majority_class=0):
    transform = T.Compose([
        T.RandomRotation(degrees=10),
    ])
    train_loader, val_loader, source_test_loader, target_test_loader = get_dataloaders(source_files, target_files,
                                                                                       batch_size,
                                                                                       source_ratio=source_ratio,
                                                                                       source_majority_class=source_majority_class, target_majority_class=target_majority_class, transform=transform)

    model = HipXrayClassifier(lr=learning_rate, filename=filename, epochs=num_epochs)

    logger = TensorBoardLogger("../DANN/tensorboard_logs/logs", name=filename)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath='',
        filename=filename,
        save_top_k=0,
        save_weights_only=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        verbose=True,
        min_delta=0.001
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback,
                   # early_stopping_callback
                   ],
        log_every_n_steps=5,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    print(filename)

    print("Source test results")
    trainer.test(model, dataloaders=source_test_loader)

    print("Target test results")
    trainer.test(model, dataloaders=target_test_loader)

    return model
