import random
from collections import defaultdict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from train_models.data_utils.data_handler import HipXrayBinaryDataset


# class ClassBalancedBatchSampler(Sampler):
#     def __init__(self, labels, batch_size, num_batches=None):
#         super().__init__()
#         self.labels = labels
#         self.batch_size = batch_size
#         self.class_indices = defaultdict(list)
#
#         for idx, label in enumerate(labels):
#             self.class_indices[label].append(idx)
#
#         self.classes = list(self.class_indices.keys())
#         self.num_classes = len(self.classes)
#         self.samples_per_class = batch_size // self.num_classes
#
#         # Determine total number of batches (optional override)
#         if num_batches is None:
#             max_class_len = max(len(v) for v in self.class_indices.values())
#             self.num_batches = (max_class_len * self.num_classes) // batch_size
#         else:
#             self.num_batches = num_batches
#
#     def __iter__(self):
#         for _ in range(self.num_batches):
#             batch = []
#             for cls in self.classes:
#                 indices = random.choices(self.class_indices[cls], k=self.samples_per_class)
#                 batch.extend(indices)
#             yield batch
#
#     def __len__(self):
#         return self.num_batches


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        super().__init__()
        self.labels = labels
        self.batch_size = batch_size
        self.class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        self.min_class_len = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = self.min_class_len * len(self.class_indices) // self.batch_size

    def __iter__(self):
        per_class = self.batch_size // len(self.class_indices)
        class_pools = {k: random.sample(v, len(v)) for k, v in self.class_indices.items()}

        for i in range(self.num_batches):
            batch = []
            for cls in self.class_indices:
                start = i * per_class
                end = start + per_class
                batch.extend(class_pools[cls][start:end])
            yield batch

    def __len__(self):
        return self.num_batches


class HipXrayDataModule(LightningDataModule):
    def __init__(self, h5_files, batch_size=32, num_workers=4, transform=None,
                 val_ratio=0.10, test_ratio=0.10, rebalance=True, seed=23, train_split='train1', val_split='val1', test_split='test1', source_majority_class=0, target_majority_class=0, source_ratio=0.5, target_ratio=0.5, same_size=True):
        super().__init__()
        self.source_majority_class = source_majority_class
        self.target_majority_class = target_majority_class
        self.source_ratio = source_ratio
        self.target_ratio = target_ratio
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rebalance = rebalance
        self.seed = seed
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.same_size = same_size

    def setup(self, stage=None):
        self.train_dataset = HipXrayBinaryDataset(
            self.h5_files,
            split=self.train_split,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.seed,
            transform=self.transform,
            source_majority_class=self.source_majority_class,
            target_majority_class=self.target_majority_class,
            rebalance=self.rebalance,
            source_ratio=self.source_ratio,
            target_ratio=self.target_ratio,
            same_size=self.same_size
        )

        self.val_dataset = HipXrayBinaryDataset(
            self.h5_files,
            split=self.val_split,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.seed,
            transform=self.transform,
            rebalance=False
        )

        self.test_dataset = HipXrayBinaryDataset(
            self.h5_files,
            split=self.test_split,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.seed,
            transform=self.transform,
            rebalance=False
        )

    def train_dataloader(self):
        labels = [self.train_dataset.labels[i] for i in self.train_dataset.indices]
        sampler = ClassBalancedBatchSampler(labels, self.batch_size)
        return DataLoader(
            self.train_dataset,
            # batch_sampler=sampler,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,)
