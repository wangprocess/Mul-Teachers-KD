import cv2
import numpy as np
import itertools
import torch
import pandas as pd
from torch import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image


class DataSet_ISIC_Teacher_or_Stu(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            df = pd.read_csv(self._base_dir + '/HQ/Train_OverSample/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'val':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_ISIC_MT_SSL(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            df = pd.read_csv(self._base_dir + '/train_merge_list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'val':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_ISIC_HQ_LQ_Test(Dataset):
    def __init__(self, base_dir=None, split='HQ-Test', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'HQ-Test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'LQ-Test':
            df = pd.read_csv(self._base_dir + '/LQ_Hairs_Aug/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_BUSI_Teacher_or_Stu(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            df = pd.read_csv(self._base_dir + '/HQ/Train_OverSample/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'val':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_BUSI_MT_SSL(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            df = pd.read_csv(self._base_dir + '/train_merge_list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'val':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_BUSI_HQ_LQ_Test(Dataset):
    def __init__(self, base_dir=None, split='HQ-Test', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'HQ-Test':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'LQ-Test':
            df = pd.read_csv(self._base_dir + '/LQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_Dermnet_Teacher_or_Stu(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            df = pd.read_csv(self._base_dir + '/HQ/Train/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'val':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_Dermnet_MT_SSL(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            df = pd.read_csv(self._base_dir + '/train_merge_list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'val':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'test':
            df = pd.read_csv(self._base_dir + '/HQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class DataSet_Dermnet_HQ_LQ_Test(Dataset):
    def __init__(self, base_dir=None, split='HQ-Test', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'HQ-Test':
            df = pd.read_csv(self._base_dir + '/HQ/Validation/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        elif self.split == 'LQ-Test':
            df = pd.read_csv(self._base_dir + '/LQ/Test/list.csv')
            for index, row in df.iterrows():
                self.sample_list.append((row.iloc[0], int(row.iloc[1])))
        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = img_path[3:]
        # 因为python解释器从运行的脚本开始算相对路径，所以要删掉../，不然找不到文件
        img = cv2.imread(img_path)
        # opencv的大坑，为了兼容老的模型使用BGR格式，使用RGB也问题不大
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # use albumentations to data augmentation
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).float()
        label = torch.tensor(label)
        return img, label


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # shuffle
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)