import json

import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import roc_curve
from torch.nn import BCEWithLogitsLoss
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
import torchvision.transforms as T

from train_models.utils.MMD_loss import MMDLoss
from train_models.data_utils.data_loaders import get_dataloaders
from train_models.utils.domain_discriminator import DomainDiscriminator
from train_models.utils.feature_extractor import FeatureExtractor
from train_models.utils.label_classifier import LabelClassifier
from itertools import cycle
import lightning as L

save_path = '../../../../../tudelft.net/staff-umbrella/MScThesisJLuu/results/'


class DomainAdaptation(L.LightningModule):
    def __init__(self, feature_extractor, label_classifier, domain_discriminator, lambda_=1.0, lr=1e-5, filename=None,
                 adaptation_type='dann'):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_classifier
        self.domain_discriminator = domain_discriminator

        self.mmd_loss = MMDLoss()

        self.lambda_ = lambda_
        self.save_hyperparameters()
        self.filename = filename

        self.label_criterion = BCEWithLogitsLoss()
        self.domain_criterion = BCEWithLogitsLoss()

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.target_auc = BinaryAUROC()
        self.source_auc = BinaryAUROC()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.domain_auc = BinaryAUROC()
        self.target_domain_auc = BinaryAUROC()
        self.val_loss_metric = MeanMetric()

        self.test_source_outputs = []
        self.test_target_outputs = []

        self.adaptation_type = adaptation_type

    def forward(self, x):
        return self.feature_extractor(x)

    def on_train_epoch_start(self):
        current_step = self.global_step
        total_steps = self.trainer.estimated_stepping_batches
        p = float(current_step) / total_steps
        self.lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

    def training_step(self, batch, batch_idx):
        (x_s, y_s), (x_t, y_t) = batch
        batch_size = x_s.size(0)
        # unique, counts = torch.unique(y_s, return_counts=True)
        # counts_dict = dict(zip(unique.tolist(), counts.tolist()))
        # print(f"Batch {batch_idx}: Class distribution {counts_dict}")
        # unique, counts = torch.unique(y_t, return_counts=True)
        # counts_dict = dict(zip(unique.tolist(), counts.tolist()))
        # print(f"Batch {batch_idx}: Class distribution {counts_dict}")
        x_s, y_s = x_s.to(self.device), y_s.float().unsqueeze(1).to(self.device)
        x_t = x_t.to(self.device)

        # Combine
        x = torch.cat([x_s, x_t], dim=0)
        domain_labels = torch.cat([
            torch.ones((x_s.size(0), 1)),
            torch.zeros((x_t.size(0), 1))
        ]).to(self.device)

        # Feature extraction
        features = self(x)

        source_features, target_features = torch.split(features, x_s.size(0))

        # Label prediction
        source_pred_labels = self.label_predictor(source_features)
        target_pred_labels = self.label_predictor(target_features)

        loss_label = self.label_criterion(source_pred_labels, y_s)

        if self.adaptation_type == 'mmd':
            domain_loss = 0.1*(self.mmd_loss(source_features, target_features) * 0.5 + self.mmd_loss(source_pred_labels,
                                                                                           target_pred_labels) * 0.5)
        else:
            # Domain prediction
            domain_preds = self.domain_discriminator(features, self.lambda_)
            domain_loss = self.domain_criterion(domain_preds, domain_labels)

            self.domain_auc.update(domain_preds, domain_labels)

        # Compute training accuracy
        with torch.no_grad():
            preds = torch.sigmoid(source_pred_labels) > 0.5
            self.train_acc.update(preds, y_s)
        loss = loss_label + domain_loss
        self.log_dict({'loss': loss, 'loss_label': loss_label, 'domain_loss': domain_loss}, prog_bar=True,
                      batch_size=batch_size)
        return loss

    def on_train_epoch_end(self):
        if self.adaptation_type == 'dann':
            auc = self.domain_auc.compute()
            self.log("domain_auc", auc, prog_bar=True, batch_size=32)
            self.domain_auc.reset()

        train_acc = self.train_acc.compute()
        self.log("train_acc", train_acc, prog_bar=True, batch_size=32)

    def validation_step(self, batch, batch_idx):
        (x_s, y_s), (_, _) = batch
        x_s, y_s = x_s.to(self.device), y_s.float().unsqueeze(1).to(self.device)

        features = self.feature_extractor(x_s)
        pred_labels = self.label_predictor(features)
        val_loss = self.label_criterion(pred_labels, y_s)

        self.val_auc.update(pred_labels, y_s.int())
        self.log("val_loss", val_loss, prog_bar=True, batch_size=32)
        return val_loss

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val_auc", auc, prog_bar=True, batch_size=32)
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        (x_s, y_s), (x_t, y_t) = batch
        x_t, y_t = x_t.to(self.device), y_t.float().unsqueeze(1).to(self.device)
        y_s = y_s.float().unsqueeze(1).to(self.device)

        # Target predictions
        features_t = self(x_t)
        logits_t = self.label_predictor(features_t)
        pred_probs_t = torch.sigmoid(logits_t)
        test_loss = self.label_criterion(logits_t, y_t)

        # Source predictions
        features_source = self(x_s)
        logits_s = self.label_predictor(features_source)
        pred_probs_s = torch.sigmoid(logits_s)

        # Domain discrimination
        x = torch.cat([x_s, x_t], dim=0)
        domain_labels = torch.cat([
            torch.ones((x_s.size(0), 1)),
            torch.zeros((x_t.size(0), 1))
        ]).to(self.device)

        features = self(x)
        domain_preds = self.domain_discriminator(features, self.lambda_)

        self.target_domain_auc.update(domain_preds, domain_labels)
        test_domain_score = self.target_domain_auc.compute()
        self.log("target_domain_auc", test_domain_score, prog_bar=True, batch_size=32)

        # Update AUCs using probabilities
        self.target_auc.update(pred_probs_t, y_t.int())
        self.source_auc.update(pred_probs_s, y_s.int())

        # Compute accuracies with threshold at 0.5
        target_acc = ((pred_probs_t > 0.5) == y_t.bool()).float().mean()
        source_acc = ((pred_probs_s > 0.5) == y_s.bool()).float().mean()
        self.log("target_acc", target_acc, prog_bar=True)
        self.log("source_acc", source_acc, prog_bar=True)

        self.test_source_outputs.append((pred_probs_s.cpu(), y_s.cpu()))
        self.test_target_outputs.append((pred_probs_t.cpu(), y_t.cpu()))

        return test_loss

    def on_test_epoch_end(self):
        target_auc_score = self.target_auc.compute()
        source_auc_score = self.source_auc.compute()
        self.log("target_auc", target_auc_score, prog_bar=True, batch_size=32)
        self.log("source_auc", source_auc_score, prog_bar=True, batch_size=32)
        self.source_auc.reset()
        self.target_auc.reset()

        # create ROC graph for source
        probs, labels = zip(*self.test_source_outputs)
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()

        fpr, tpr, thresholds = roc_curve(labels, probs)
        # Save ROC values for later comparison
        source_roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(source_auc_score),
        }

        # create ROC graph for target
        probs, labels = zip(*self.test_target_outputs)
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()

        fpr, tpr, thresholds = roc_curve(labels, probs)
        # Save ROC values for later comparison
        target_roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(target_auc_score),
        }
        with open(f"{save_path}graphs/source_{self.filename}_roc_values.json", "w") as f:
            json.dump(source_roc_data, f)
        with open(f"{save_path}graphs/target_{self.filename}_roc_values.json", "w") as f:
            json.dump(target_roc_data, f)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,  # Decay LR every 10 epochs
            gamma=0.1  # Multiply LR by 0.1
        )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     # 'scheduler': scheduler,
            #     'interval': 'epoch',
            #     'frequency': 1,
            # }
        }


def train_model(source, target, filename, num_epochs=100, lr=0.0001, adaptation_type='dann', source_majority_class=0, target_majority_class=0, source_ratio=0.5, target_ratio=0.5, same_size=True):
    transform = T.Compose([
        T.RandomRotation(degrees=10),
    ])

    train_dataloader, val_dataloader, source_test, target_test = get_dataloaders(source, target, transform=transform, source_majority_class=source_majority_class, target_majority_class=target_majority_class, source_ratio=source_ratio, target_ratio=target_ratio, same_size=same_size)

    model = DomainAdaptation(feature_extractor=FeatureExtractor(), label_classifier=LabelClassifier(input_dim=512),
                             domain_discriminator=DomainDiscriminator(input_dim=512), lambda_=1.0, lr=lr,
                             filename=filename, adaptation_type=adaptation_type)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath='',
        filename=filename,
        save_top_k=0,
        save_weights_only=True,
        verbose=True,
        monitor='val_auc',
        mode='max'
    )
    logger = TensorBoardLogger("/logs", name=filename)

    # Trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        # logger=logger
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.test(model, dataloaders=source_test)

    print(filename)
