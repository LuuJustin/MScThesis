import json

import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryAccuracy

from torchvision.models import resnet18, ResNet18_Weights

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

import torchvision.transforms as T
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train_models.data_utils.data_loaders import get_dataloaders
from train_models.utils.feature_extractor import FeatureExtractor
from train_models.utils.MMD_loss import MMDLoss
from data_utils.data_handler import HipXrayBinaryDataset

from MScThesis.domain_adaptation.train_models.utils.label_classifier import LabelClassifier

# save model
save_path = '../../../../../../../tudelft.net/staff-umbrella/MScThesisJLuu/models'
model_path = '/home/nfs/jluu/model_code'

N_tsne_samples = 500


def plot_tsne(feat_s, feat_t, y_s, y_t, epoch, save_dir=save_path, filename=None):
    feat_s = feat_s.detach().cpu().numpy()
    feat_t = feat_t.detach().cpu().numpy()
    y_s = y_s.detach().cpu().numpy().squeeze()
    y_t = y_t.detach().cpu().numpy().squeeze()

    # Stack features and labels
    feats = np.vstack([feat_s, feat_t])
    labels = np.concatenate([y_s, y_t])
    domains = np.array(["Source"] * len(feat_s) + ["Target"] * len(feat_t))

    # Run PCA + t-SNE
    X_pca = PCA(n_components=40, random_state=42).fit_transform(feats)
    tsne_feats = TSNE(n_components=2, perplexity=30, learning_rate=10, n_iter=1000, random_state=42).fit_transform(
        X_pca)

    # DataFrame for plotting
    df = pd.DataFrame({
        "x": tsne_feats[:, 0],
        "y": tsne_feats[:, 1],
        "Domain": domains,
        "Class": labels.astype(int)
    })

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue="Domain", style="Class",
        palette={"Source": "blue", "Target": "red"},
        alpha=0.7
    )
    plt.title(f"t-SNE at Epoch {epoch}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir + f"/{filename}_tsne_epoch{epoch}.png")
    plt.close()


class HipXrayClassifier(L.LightningModule):
    def __init__(self, epochs=100, labda=1.0, lr=5e-5, filename=None, weight_decay=1e-4, dropout_prob=0.6,
                 use_coral=False):
        super().__init__()
        self.weight_decay = weight_decay
        self.lr = lr
        self.mmd_loss = MMDLoss(kernel_num=5)

        self.save_hyperparameters()
        self.use_coral = use_coral

        self.feature_extractor = FeatureExtractor()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = LabelClassifier(input_dim=512)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.MMD = MMDLoss()
        self.labda = labda

        # metrics
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.target_auc = BinaryAUROC()
        self.source_auc = BinaryAUROC()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.target_domain_auc = BinaryAUROC()
        self.val_loss_metric = MeanMetric()

        self.val_loss_metric = MeanMetric()
        self.filename = filename

        self.test_source_outputs = []
        self.test_target_outputs = []

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def on_train_epoch_start(self):
        progress = self.current_epoch / self.trainer.max_epochs
        scaled = progress ** 0.5
        self.labda = self.hparams.labda * scaled

    def training_step(self, batch, batch_idx):
        (x_s, y_s), (x_t, _) = batch

        y_logits_s, feat_s = self(x_s)
        y_logits_t, feat_t = self(x_t)

        y_s = y_s.float().unsqueeze(1)

        cls_loss = self.bce_loss(y_logits_s, y_s)

        mmd_loss = self.MMD(x_s, x_t) * 0.5 + self.MMD(y_logits_s, y_logits_t) * 0.5

        total_loss = cls_loss + mmd_loss

        self.log("mmd_loss", mmd_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log("train_loss", total_loss, prog_bar=False)
        self.log("cls_loss", cls_loss, prog_bar=True)

        # Calculate predictions
        y_probs = torch.sigmoid(y_logits_s)
        y_preds = (y_probs > 0.5).float()

        # Update metrics
        self.train_acc.update(y_preds, y_s)
        self.train_auc.update(y_probs, y_s)
        return total_loss

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
        logits, _ = self(x)
        probs = torch.sigmoid(logits).squeeze(1)

        y = y.float().unsqueeze(1)

        # Calculate loss
        loss = self.bce_loss(logits, y)

        self.val_auc.update(probs, y)
        self.val_acc.update((probs > 0.5).float().unsqueeze(1), y)
        self.val_loss_metric.update(loss)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_auc = self.val_auc.compute()
        # self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_auc", val_auc, prog_bar=True)
        self.val_acc.reset()
        self.val_auc.reset()

        avg_loss = self.val_loss_metric.compute()
        self.log("val_loss", avg_loss, prog_bar=False)
        self.val_loss_metric.reset()

    def on_test_start(self):
        self.test_outputs = []

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
        with open(f"graphs/source_{self.filename}_roc_values.json", "w") as f:
            json.dump(source_roc_data, f)
        with open(f"graphs/target_{self.filename}_roc_values.json", "w") as f:
            json.dump(target_roc_data, f)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[15], gamma=0.5
        # )
        # return [optimizer], [scheduler]
        return optimizer

def train_model(source, target, filename, num_epochs=100,
                lr=1e-4, labda=1.0, source_ratio=0.5, target_ratio=0.5, batch_size=32,
                source_majority_class=0, target_majority_class=0, use_coral=False, rebalance=True):
    transform = T.Compose([
        T.RandomRotation(degrees=10),
    ])
    train_dataloader, val_dataloader, source_test, target_test = get_dataloaders(source, target, transform=transform)

    model = HipXrayClassifier(epochs=num_epochs, lr=lr, labda=labda, filename=filename, use_coral=use_coral)

    logger = TensorBoardLogger("../../tensorboard_logs/logs", name=filename)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath='',
        filename=filename,
        save_top_k=0,
        save_weights_only=True,
        verbose=True,
        monitor='val_auc',
        mode='max'
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        logger=logger
    )

    # # plot tsne before training
    # print("plotting tsne before")
    # plot_feat_s, plot_y_s, plot_feat_t, plot_y_t = [], [], [], []
    # with torch.no_grad():
    #     for batch in tsne_loader:
    #         (x_s, y_s_batch), (x_t, y_t_batch) = batch
    #         _, f_s = model(x_s.to(device='cpu'))
    #         _, f_t = model(x_t.to(device='cpu'))
    #
    #         plot_feat_s.append(f_s.cpu())
    #         plot_feat_t.append(f_t.cpu())
    #         plot_y_s.append(y_s_batch.cpu())
    #         plot_y_t.append(y_t_batch.cpu())
    #
    # plot_feat_s = torch.cat(plot_feat_s)[:N_tsne_samples]
    # plot_feat_t = torch.cat(plot_feat_t)[:N_tsne_samples]
    # plot_y_s = torch.cat(plot_y_s)[:N_tsne_samples].float().unsqueeze(1)
    # plot_y_t = torch.cat(plot_y_t)[:N_tsne_samples].float().unsqueeze(1)
    #
    # plot_tsne(plot_feat_s, plot_feat_t, plot_y_s, plot_y_t, epoch=0, filename=filename)

    trainer.fit(model, train_dataloader, val_dataloader)

    # # plot tsne after training
    # print("plotting tsne after")
    #
    # plot_feat_s, plot_y_s, plot_feat_t, plot_y_t = [], [], [], []
    # with torch.no_grad():
    #     for batch in tsne_loader:
    #         (x_s, y_s_batch), (x_t, y_t_batch) = batch
    #         _, f_s = model(x_s.to(device='cpu'))
    #         _, f_t = model(x_t.to(device='cpu'))
    #
    #         plot_feat_s.append(f_s.cpu())
    #         plot_feat_t.append(f_t.cpu())
    #         plot_y_s.append(y_s_batch.cpu())
    #         plot_y_t.append(y_t_batch.cpu())
    #
    # plot_feat_s = torch.cat(plot_feat_s)[:N_tsne_samples]
    # plot_feat_t = torch.cat(plot_feat_t)[:N_tsne_samples]
    # plot_y_s = torch.cat(plot_y_s)[:N_tsne_samples].float().unsqueeze(1)
    # plot_y_t = torch.cat(plot_y_t)[:N_tsne_samples].float().unsqueeze(1)
    #
    # plot_tsne(plot_feat_s, plot_feat_t, plot_y_s, plot_y_t, epoch=num_epochs, filename=filename)
    print(filename)
    trainer.test(model, dataloaders=source_test)


    return model
