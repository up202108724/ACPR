from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random

def get_dataloaders(root_dir, transform, batch_size=64, num_workers=4):
    dataset = ImageFolder(root=root_dir, transform=transform)
    class_to_idx = dataset.class_to_idx
    real_label = class_to_idx['real']
    fake_label = class_to_idx['fake']

    # Balance
    real_idx = [i for i, (_, lbl) in enumerate(dataset) if lbl == real_label]
    fake_idx = [i for i, (_, lbl) in enumerate(dataset) if lbl == fake_label]
    random.shuffle(real_idx), random.shuffle(fake_idx)
    real_idx, fake_idx = real_idx[:30000], fake_idx[:30000]

    def split(indices):
        total = len(indices)
        train, val = int(0.7 * total), int(0.15 * total)
        return indices[:train], indices[train:train+val], indices[train+val:]

    r_train, r_val, r_test = split(real_idx)
    f_train, f_val, f_test = split(fake_idx)

    train_idx = r_train + f_train
    val_idx = r_val + f_val
    test_idx = r_test + f_test
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader