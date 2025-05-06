from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class HipXrayBinaryDataset(Dataset):
    def __init__(self, h5_files, split='train', val_ratio=0.15, test_ratio=0.15, random_seed=42):
        assert split in {'train', 'val', 'test'}, "split must be 'train', 'val', or 'test'"
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

        # Group image indices by subject
        subject_to_indices = {}
        for i, sid in enumerate(self.subject_ids):
            subject_to_indices.setdefault(sid, []).append(i)

        all_subjects = np.array(list(subject_to_indices.keys()))
        stratify_labels = [int(np.any(self.labels[subject_to_indices[s]] == 1)) for s in all_subjects]
        print(stratify_labels)
        # train_val vs test splits
        train_val_subjects, test_subjects = train_test_split(
            all_subjects,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=stratify_labels
        )

        # train vs val splits
        train_val_labels = [int(np.any(self.labels[subject_to_indices[s]] == 1)) for s in train_val_subjects]
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=val_ratio / (1 - test_ratio),
            random_state=random_seed,
            stratify=train_val_labels
        )

        split_map = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

        selected_subjects = split_map[split]
        self.indices = [i for s in selected_subjects for i in subject_to_indices[s]]

        selected_labels = self.labels[self.indices]
        print(f"[{split.upper()}] Loaded {len(self.indices)} samples ({len(selected_subjects)} subjects). "
              f"Class distribution: {np.bincount(selected_labels)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = torch.tensor(self.images[real_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return img, label
