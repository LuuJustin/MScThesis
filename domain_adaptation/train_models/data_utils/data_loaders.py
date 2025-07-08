from itertools import cycle

from train_models.data_utils.data_module import HipXrayDataModule


class CombinedDataLoader:
    def __init__(self, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.length = len(source_loader)

    def __iter__(self):
        self.source_iter = iter(self.source_loader)
        self.target_iter = cycle(self.target_loader)
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        source_batch = next(self.source_iter)
        target_batch = next(self.target_iter)
        return source_batch, target_batch


def get_dataloaders(source, target, transform=None, source_majority_class=0, target_majority_class=0, source_ratio=0.5,
                    target_ratio=0.5, same_size=True):
    source_datamodule = HipXrayDataModule(source, train_split='train1', test_split='test1', val_split='val1',
                                          source_majority_class=source_majority_class,
                                          target_majority_class=target_majority_class, source_ratio=source_ratio,
                                          target_ratio=target_ratio, transform=transform, same_size=same_size)
    target_datamodule = HipXrayDataModule(source, train_split='train2', test_split='test2', val_split='val2',
                                          source_majority_class=source_majority_class,
                                          target_majority_class=target_majority_class, source_ratio=source_ratio,
                                          target_ratio=target_ratio, same_size=same_size)

    source_datamodule.setup()
    target_datamodule.setup()

    source_train_loader = source_datamodule.train_dataloader()
    target_train_loader = target_datamodule.train_dataloader()

    train_dataloader = CombinedDataLoader(source_train_loader, target_train_loader)

    source_val_loader = source_datamodule.val_dataloader()
    target_val_loader = target_datamodule.val_dataloader()

    val_dataloader = CombinedDataLoader(source_val_loader, target_val_loader)

    source_test_loader = source_datamodule.test_dataloader()
    target_test_loader = target_datamodule.test_dataloader()

    source_test_loader = CombinedDataLoader(source_test_loader, target_test_loader)

    return train_dataloader, val_dataloader, source_test_loader, target_test_loader
