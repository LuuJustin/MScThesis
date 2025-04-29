from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

class HipXrayBinaryDataset(Dataset):
    def __init__(self, h5_files, split='train', val_ratio=0.2, random_seed=42):
        """
        :param h5_files: list of paths to .h5 files (OAI or CHECK)
        :param split: 'train' or 'val'
        """
        self.images = []
        self.labels = []

        for path in h5_files:
            with h5py.File(path, "r") as f:
                for side in ["left_hip", "right_hip"]:
                    imgs = f[f"{side}/images"][:]
                    scores = f[f"{side}/scores"][:]

                    # Relabel: 0–1 -> 0, 2–4 -> 1
                    labels = np.where(scores <= 1, 0, 1)

                    # Flip left hips horizontally
                    if side == "left_hip":
                        imgs = np.flip(imgs, axis=-1)

                    self.images.append(imgs)
                    self.labels.append(labels)

        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

        # Split into train/val
        train_idx, val_idx = train_test_split(
            np.arange(len(self.labels)),
            test_size=val_ratio,
            stratify=self.labels,
            random_state=random_seed
        )
        self.indices = train_idx if split == 'train' else val_idx
        print(f"[{split.upper()}] Loaded {len(self.indices)} samples from {len(h5_files)} HDF5 files.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = torch.tensor(self.images[real_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return img, label
