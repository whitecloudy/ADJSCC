import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR10


class CIFAR10CustomDataset(Dataset):
    def __init__(self, dataset_base, snr_db, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        self.dataset_base = dataset_base
        self.snr_db = torch.from_numpy(np.array(snr_db, dtype=np.float32)).unsqueeze(1)
        self.h_real = torch.from_numpy(np.array(h_real, dtype=np.float32)).unsqueeze(1) if h_real is not None else None
        self.h_imag = torch.from_numpy(np.array(h_imag, dtype=np.float32)).unsqueeze(1) if h_imag is not None else None
        self.b_prob = torch.from_numpy(np.array(b_prob, dtype=np.float32)).unsqueeze(1) if b_prob is not None else None
        self.b_stddev = torch.from_numpy(np.array(b_stddev, dtype=np.float32)).unsqueeze(1) if b_stddev is not None else None

    def __len__(self):
        return len(self.dataset_base)

    def __getitem__(self, idx):
        item = self.dataset_base[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item

        inputs = [img, self.snr_db[idx]]
        if self.h_real is not None and self.h_imag is not None:
            inputs.extend([self.h_real[idx], self.h_imag[idx]])
        if self.b_prob is not None and self.b_stddev is not None:
            inputs.extend([self.b_prob[idx], self.b_stddev[idx]])

        target = img * 255.0
        return inputs, target


def _load_cifar10():
    if os.path.exists('./data/cifar10/train') and os.path.exists('./data/cifar10/test'):
        train_base = ImageFolder('./data/cifar10/train', transform=T.ToTensor())
        test_base = ImageFolder('./data/cifar10/test', transform=T.ToTensor())
    else:
        transform = T.ToTensor()
        train_base = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_base = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_base, test_base


def get_dataset_snr(snr_db):
    train_base, test_base = _load_cifar10()
    train_nums, test_nums = len(train_base), len(test_base)
    snrdb_train = snr_db * np.ones(shape=(train_nums,))
    snrdb_test = snr_db * np.ones(shape=(test_nums,))

    train_ds = CIFAR10CustomDataset(train_base, snrdb_train)
    test_ds = CIFAR10CustomDataset(test_base, snrdb_test)
    return (train_ds, train_nums), (test_ds, test_nums)


def get_dataset_snr_and_h(snr_db):
    train_base, test_base = _load_cifar10()
    train_nums, test_nums = len(train_base), len(test_base)
    snrdb_train = snr_db * np.ones(shape=(train_nums,))
    h_real_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    h_imag_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )

    snrdb_test = snr_db * np.ones(shape=(test_nums,))
    h_real_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )
    h_imag_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )

    train_ds = CIFAR10CustomDataset(train_base, snrdb_train, h_real_train, h_imag_train)
    test_ds = CIFAR10CustomDataset(test_base, snrdb_test, h_real_test, h_imag_test)
    return (train_ds, train_nums), (test_ds, test_nums)


def get_dataset_snr_range(snr_db_low, snr_db_high):
    train_base, test_base = _load_cifar10()
    train_nums, test_nums = len(train_base), len(test_base)
    snrdb_train = np.random.rand(train_nums, ) * (snr_db_high - snr_db_low) + snr_db_low
    snrdb_test = np.random.rand(test_nums, ) * (snr_db_high - snr_db_low) + snr_db_low

    train_ds = CIFAR10CustomDataset(train_base, snrdb_train)
    test_ds = CIFAR10CustomDataset(test_base, snrdb_test)
    return (train_ds, train_nums), (test_ds, test_nums)


def get_dataset_snr_range_and_h(snr_db_low, snr_db_high):
    train_base, test_base = _load_cifar10()
    train_nums, test_nums = len(train_base), len(test_base)
    h_real_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    h_imag_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    snrdb_train = np.random.rand(train_nums, ) * (snr_db_high - snr_db_low) + snr_db_low

    snrdb_test = np.random.rand(test_nums, ) * (snr_db_high - snr_db_low) + snr_db_low
    h_real_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )
    h_imag_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )

    train_ds = CIFAR10CustomDataset(train_base, snrdb_train, h_real_train, h_imag_train)
    test_ds = CIFAR10CustomDataset(test_base, snrdb_test, h_real_test, h_imag_test)
    return (train_ds, train_nums), (test_ds, test_nums)


def get_test_dataset_burst(snr_db, b_prob, b_stddev):
    _, test_base = _load_cifar10()
    test_nums = len(test_base)
    snrdb_test = snr_db * np.ones(shape=(test_nums,))
    b_prob_test = b_prob * np.ones(shape=(test_nums,))
    b_stddev_test = b_stddev * np.ones(shape=(test_nums,))

    test_ds = CIFAR10CustomDataset(test_base, snrdb_test, b_prob=b_prob_test, b_stddev=b_stddev_test)
    return test_ds, test_nums
