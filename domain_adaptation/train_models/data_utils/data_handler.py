import random

import h5py
import numpy as np
import torch
from collections import defaultdict, Counter

from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import resample
from torch.utils.data import Dataset

random_seed = 23


def normalize_per_image(img):
    """
    img: NumPy array of shape (C, H, W), dtype float32 expected.
    Normalize each image by subtracting its mean and dividing by its std dev.
    """
    mean = img.mean()
    std = img.std()
    if std < 1e-6:
        std = 1.0  # to avoid division by zero
    return (img - mean) / std


def add_poisson_noise(img):
    noisy = np.random.poisson(img.astype(np.float32))
    return np.clip(noisy, 0, 255).astype(np.float32)


def add_blur(image, sigma):
    return gaussian_filter(image, sigma=sigma)

def flip_horizontal(img):
    return np.flip(img, axis=2)  # horizontal flip


def adjust_brightness(img, factor):
    """img: NumPy array (3, H, W), factor >1 brightens, <1 darkens"""
    img = img.astype(np.float32) * factor
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def stratified_subject_split(subject_ids, subject_labels, test_size=0.5, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(subject_ids, subject_labels))

    train_subjects = subject_ids[train_idx]
    train2_subjects = subject_ids[test_idx]

    return train_subjects, train2_subjects


def split_subject_group(subject_ids, subject_to_labels, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    subject_labels = np.array([1 if np.any(subject_to_labels[sid]) else 0 for sid in subject_ids])

    trainval_ids, test_ids, trainval_labels, _ = train_test_split(
        subject_ids, subject_labels,
        test_size=test_ratio,
        stratify=subject_labels,
        random_state=random_seed
    )

    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    train_ids, val_ids, _, _ = train_test_split(
        trainval_ids, trainval_labels,
        test_size=val_ratio_adjusted,
        stratify=trainval_labels,
        random_state=random_seed
    )

    return train_ids, val_ids, test_ids


class HipXrayBinaryDataset(Dataset):
    def __init__(self, h5_files, split='train', val_ratio=0.10, test_ratio=0.10, random_seed=random_seed,
                 transform=None,
                 source_ratio=0.5, target_ratio=0.5, source_majority_class=0, target_majority_class=0, rebalance=True, same_size=True):
        assert split in ['train1', 'val1', 'test1', 'train2', 'val2', 'test2']
        # assert split in ['train', 'val', 'test', ]

        self.transform = transform
        self.images = []
        self.labels = []
        self.subject_ids = []
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.same_size = same_size

        for path in h5_files:
            with h5py.File(path, "r") as f:
                for side in ["left_hip", "right_hip"]:
                    imgs = f[f"{side}/images"][:]
                    scores = f[f"{side}/scores"][:]
                    sids = f[f"{side}/subject_ids"][:].astype(str)

                    # Binarize labels
                    labels = np.where(scores <= 1, 0, 1)

                    # Flip left hip horizontally
                    if side == "left_hip":
                        imgs = np.flip(imgs, axis=-1)

                    self.images.append(imgs)
                    self.labels.append(labels)
                    self.subject_ids.append(sids)

        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)
        self.subject_ids = np.concatenate(self.subject_ids)

        # Group indices and assign subject-level labels ---
        subject_to_indices = defaultdict(list)
        subject_to_labels = defaultdict(list)

        for i, sid in enumerate(self.subject_ids):
            subject_to_indices[sid].append(i)
            subject_to_labels[sid].append(self.labels[i])

        print("number of images", len(self.images))
        print("number of labels", len(self.labels))
        # subject-level labels for stratification
        subject_ids = np.array(list(subject_to_indices.keys()))
        subject_labels = np.array([1 if np.any(subject_to_labels[sid]) else 0 for sid in subject_ids])

        # Stratified split at subject level
        # train_subjects, test_subjects = stratified_subject_split(subject_ids, subject_labels, test_size=0.1)

        train1_ids, val1_ids, test1_ids = split_subject_group(subject_ids, subject_to_labels)
        train2_ids, val2_ids, test2_ids = split_subject_group(subject_ids, subject_to_labels)

        # train1_ids, val1_ids, test1_ids = split_subject_group(train_subjects, subject_to_labels)
        # train2_ids, val2_ids, test2_ids = split_subject_group(test_subjects, subject_to_labels)

        # subject IDs for the split
        if split == 'train1':
            chosen_sids = train1_ids
        elif split == 'val1':
            chosen_sids = val1_ids
        elif split == 'test1':
            chosen_sids = test1_ids

        if split == 'train2':
            chosen_sids = train2_ids
        elif split == 'val2':
            chosen_sids = val2_ids
        elif split == 'test2':
            chosen_sids = test2_ids

        # if split == 'train':
        #     chosen_sids = train_sids
        # elif split == 'val':
        #     chosen_sids = val_sids
        # elif split == 'test':
        #     chosen_sids = test_sids

        self.indices = [i for sid in chosen_sids for i in subject_to_indices[sid]]

        if split == 'train1' and rebalance:
            # self.indices = self.rebalance_indices(self.indices, ratio=ratio, seed=random_seed,
            #                                       majority_class=majority_class)
            print(self.same_size)
            if self.same_size:
                self.indices = self.rebalance_indices_fixed_size(self.indices, 1936, source_ratio, majority_class=source_majority_class,
                                                             seed=random_seed)
            else:
                self.indices = self.rebalance_indices(self.indices, ratio=source_ratio, majority_class=source_majority_class, seed=random_seed)
            print(f"[{split.upper()}] Loaded {len(self.indices)} samples from {len(h5_files)} files. "
                  f"Patients: {len(train1_ids)}.")

        if split == 'train2' and rebalance:
            # self.indices = self.rebalance_indices(self.indices, ratio=ratio, seed=random_seed,
            #                                       majority_class=majority_class)
            if self.same_size:
                self.indices = self.rebalance_indices_fixed_size(self.indices, 1936, target_ratio, majority_class=target_majority_class, seed=random_seed)
            else:
                self.indices = self.rebalance_indices(self.indices, ratio=target_ratio, majority_class=target_majority_class, seed=random_seed)

            print(f"[{split.upper()}] Loaded {len(self.indices)} samples from {len(h5_files)} files. "
                  f"Patients: {len(train2_ids)}.")

        if split.endswith('2'):
            for i in self.indices:
                img = self.images[i]
                self.images[i] = np.array(add_blur(adjust_brightness(img, 1.05), 1.75), np.float32)

        # if ratio and split == 'train' and rebalance:
        #     self.indices = self.rebalance_indices(self.indices, ratio=ratio, seed=random_seed,
        #                                           majority_class=majority_class)

        # Print class ratio
        subset_labels = self.labels[self.indices]
        unique, counts = np.unique(subset_labels, return_counts=True)
        total = len(subset_labels)
        ratios = {int(cls): round(count / total, 3) for cls, count in zip(unique, counts)}
        counts_dict = {int(cls): count for cls, count in zip(unique, counts)}

        print(f"  Class distribution: {ratios} (counts: {counts_dict})")

    def rebalance_indices_fixed_size(self, indices, total_size, ratio=0.5, majority_class=0, seed=42):
        """
        Rebalance a fixed number of indices to match a desired class ratio.

        Args:
            indices (list): List of image indices belonging to the current split.
            total_size (int): Total number of samples to return.
            ratio (float): Desired proportion of the majority class.
            majority_class (int): Label of the majority class (usually 0).
            seed (int): Random seed for reproducibility.

        Returns:
            list: Rebalanced list of indices.
        """
        rng = np.random.default_rng(seed)

        # Split indices by class
        maj_class_indices = [idx for idx in indices if self.labels[idx] == majority_class]
        min_class_indices = [idx for idx in indices if self.labels[idx] != majority_class]

        # Compute desired counts
        n_maj = int(total_size * ratio)
        n_min = total_size - n_maj

        if len(maj_class_indices) < n_maj or len(min_class_indices) < n_min:
            raise ValueError("Not enough data to satisfy the rebalancing requirements.")

        # Sample and combine
        sampled_maj = rng.choice(maj_class_indices, size=n_maj, replace=False)
        sampled_min = rng.choice(min_class_indices, size=n_min, replace=False)

        balanced_indices = np.concatenate([sampled_maj, sampled_min])
        rng.shuffle(balanced_indices)

        return balanced_indices.tolist()

    def rebalance_indices(self, indices, ratio=0.5, seed=random_seed, tolerance=0.01, majority_class=0):
        print("Majority class: ", majority_class)

        # class distribution BEFORE rebalancing
        original_counts = Counter([self.labels[i] for i in indices])
        total_before = sum(original_counts.values())
        print(f"Before rebalancing: Class 0 = {original_counts.get(0, 0)}, Class 1 = {original_counts.get(1, 0)}")
        print(f"Original ratio of class {majority_class}: {original_counts.get(majority_class, 0) / total_before:.4f}")

        # Separate class indices
        class0 = [i for i in indices if self.labels[i] == 0]
        class1 = [i for i in indices if self.labels[i] == 1]

        if majority_class == 0:
            majority_indices, minority_indices = class0, class1
            current_ratio = float(len(majority_indices) / len(indices))
            if current_ratio < ratio:
                n_min_samples = len(minority_indices) - int(
                    abs(len(majority_indices) * (1 - ratio) - len(minority_indices) * ratio) / ratio)
                final_indices = majority_indices + resample(minority_indices, replace=False, n_samples=n_min_samples,
                                                            random_state=seed)
            else:
                n_maj_samples = int(len(minority_indices) / (1 - ratio)) - len(minority_indices)
                final_indices = resample(majority_indices, replace=False, n_samples=n_maj_samples,
                                         random_state=seed) + minority_indices
        else:
            majority_indices, minority_indices = class1, class0
            n_min_samples = int(len(majority_indices) / ratio) - len(majority_indices)
            final_indices = resample(minority_indices, replace=False, n_samples=n_min_samples,
                                     random_state=seed) + majority_indices

        # --- Print class distribution AFTER rebalancing ---
        counts = Counter([self.labels[i] for i in final_indices])
        achieved_ratio = counts.get(majority_class, 0) / len(final_indices)

        # print(f"After rebalancing: Class 0 = {counts.get(0, 0)}, Class 1 = {counts.get(1, 0)}")
        # print(f"Achieved ratio of class {majority_class}: {achieved_ratio:.4f}")

        return final_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = torch.tensor(self.images[real_idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return img, label
