import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from data_handler import HipXrayBinaryDataset


save_path = '../../../../tudelft.net/staff-umbrella/MScThesisJLuu/models'


def get_dataloaders(oai_files, check_files=None, batch_size=32):
    train_ds = HipXrayBinaryDataset(oai_files, split='train')
    val_ds = HipXrayBinaryDataset(oai_files, split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if check_files:
        check_val = HipXrayBinaryDataset(check_files, split='val')  # test-on-CHECK
        check_val_loader = DataLoader(check_val, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, check_val_loader

    return train_loader, val_loader, None


def get_model():
    model = models.resnet18(weights=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output (logits)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    preds, targets = [], []

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.size(0)
        preds.extend(torch.sigmoid(out).detach().cpu().numpy())
        targets.extend(y.cpu().numpy())

    preds = (np.array(preds) > 0.5).astype(int)
    acc = accuracy_score(targets, preds)
    return epoch_loss / len(loader.dataset), acc


def evaluate(model, loader, criterion, device, desc="Validation"):
    model.eval()
    epoch_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            x, y = x.to(device), y.float().to(device).unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss.item() * x.size(0)

            preds.extend(torch.sigmoid(out).cpu().numpy())
            targets.extend(y.cpu().numpy())

    preds_bin = (np.array(preds) > 0.5).astype(int)
    acc = accuracy_score(targets, preds_bin)
    auc = roc_auc_score(targets, preds)
    return epoch_loss / len(loader.dataset), acc, auc


def train_model(oai_files, check_files=None, num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    train_loader, val_loader, check_val_loader = get_dataloaders(oai_files, check_files)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        save_model(model, f'{save_path}/model_epoch_{epoch}.pth')

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        if check_val_loader:
            chk_loss, chk_acc, chk_auc = evaluate(model, check_val_loader, criterion, device, desc="CHECK Eval")
            print(f"CHECK Eval: Loss: {chk_loss:.4f} | Acc: {chk_acc:.4f} | AUC: {chk_auc:.4f}")

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
