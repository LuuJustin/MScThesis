from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class HipXrayBinaryDataset(Dataset):
    def __init__(self, h5_files, split='train', val_ratio=0.2, random_seed=42):
        self.images = []
        self.labels = []
        self.subject_ids = []

        for path in h5_files:
            with h5py.File(path, "r") as f:
                for side in ["left_hip", "right_hip"]:
                    imgs = f[f"{side}/images"][:]
                    scores = f[f"{side}/scores"][:]
                    subjects = f[f"{side}/subject_ids"][:]

                    labels = np.where(scores <= 1, 0, 1)

                    if side == "left_hip":
                        imgs = np.flip(imgs, axis=-1)

                    self.images.append(imgs)
                    self.labels.append(labels)
                    self.subject_ids.append(subjects)

        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)
        self.subject_ids = np.concatenate(self.subject_ids)

        # Group indices by subject
        subject_to_indices = {}
        for i, sid in enumerate(self.subject_ids):
            subject_to_indices.setdefault(sid, []).append(i)

        all_subjects = np.array(list(subject_to_indices.keys()))
        train_subjects, val_subjects = train_test_split(
            all_subjects,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=[self.labels[subject_to_indices[s][0]] for s in all_subjects]
        )

        selected_subjects = train_subjects if split == 'train' else val_subjects
        self.indices = [i for s in selected_subjects for i in subject_to_indices[s]]

        print(f"[{split.upper()}] Loaded {len(self.indices)} samples from {len(h5_files)} HDF5 files ({len(selected_subjects)} subjects).")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = torch.tensor(self.images[real_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return img, label

