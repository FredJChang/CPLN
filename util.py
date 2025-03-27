# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import itertools
import pandas as pd

from torch.nn import functional as F
# import networks
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

class WarmupCosineLrScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))            
            #ratio = 0.5 * (1. + np.cos(np.pi * real_iter / real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio

def most_common_value_with_threshold(lst, threshold):
    counter = Counter(lst)
    total_count = len(lst)

    for value, count in counter.items():
        if count / total_count > threshold:
            return value

    return None

def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + '-' + timestampTime

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.extend([ param_group['lr'] ])
    return lr


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
        # primary_iter = iterate_once(self.primary_indices)
        # secondary_iter = iterate_once(self.secondary_indices)
        # primary_batches = grouper(primary_iter, self.primary_batch_size)
        # secondary_batches = grouper(secondary_iter, self.secondary_batch_size)
        # return (
        #     primary_batch + secondary_batch
        #     for (primary_batch, secondary_batch)
        #     in zip(primary_batches, secondary_batches)
        # )
        
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



def balance_data_list(data, label):
    # 将数据和标签转换为DataFrame
    df = pd.DataFrame({'data': data, 'label': label})
    
    # 统计每个类别的样本数量
    label_counts = df['label'].value_counts()
    
    # 找出样本数量最多的类别
    max_num_class = label_counts.max()
    
    # 初始化列表来存储重采样后的数据
    balanced_data = []
    
    # 对每个类别进行过采样
    for cls in label_counts.index:
        cls_data = df[df['label'] == cls]
        repeat_factor = int(np.round(max_num_class / label_counts[cls]))
        balanced_data.append(cls_data.sample(n=max_num_class, replace=True).reset_index(drop=True))
    
    # 合并所有类别的重采样数据
    balanced_df = pd.concat(balanced_data).reset_index(drop=True)
    
    # 提取平衡后的数据和标签
    balanced_data_list = balanced_df['data'].tolist()
    balanced_label_list = balanced_df['label'].tolist()
    
    return balanced_data_list, balanced_label_list




def balance_data(data, class_num):
    data_start_index = data.index.min()
    label_one_hot = data.iloc[:,1:-1].values.astype(int)
    label = np.argmax(label_one_hot, axis=1)
    
    idx, num_class_per, roundn, file_imp = [], [], [], []
    for t in range(class_num):
        idx.append(np.where(label==t)[0])
    for itm in idx:
        num_class_per.append(len(itm))
    #num_class  [154, 937, 71, 46, 157, 16, 19] 20% 1400个   [53, 205, 20, 11, 52, 4, 5] 5% 350个 1443
    max_num_class = np.max(num_class_per)
    for i in range(len(num_class_per)):
        roundn.append(np.round(max_num_class/num_class_per[i]))
    for i in range(len(roundn)):
        for j in range(int(roundn[i])):
            file_imp.append(data.iloc[idx[i]])
    data = pd.concat(file_imp)
    return data, num_class_per

def mixup_tru_unc(true, unc, true_label, unc_label, alpha=32.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    true_size = true.size()[0]
    unc_size = unc.size()[0]    
    if true_size > unc_size:
        index = torch.randperm(true.size()[0])[:unc.size()[0]].cuda()
        mixed_x = lam * unc + (1 - lam) * true[index, :]
        y_a, y_b = unc_label, true_label[index]

    else:
        index = torch.randperm(unc.size()[0])[:true.size()[0]].cuda()
        mixed_x = lam * unc[index, :] + (1 - lam) * true
        y_a, y_b = unc_label[index], true_label


    return mixed_x, y_a, y_b, lam


def mixup_data(x, y, alpha=32.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_label = lam * y + (1 - lam) * y[index]
    return mixed_x, y_a, y_b, lam, mixed_label
    
def mixup_criterion(pred, y_a, y_b, lam):    
    return lam * F.cross_entropy(pred, y_a.float()) + (1 - lam) * F.cross_entropy(pred, y_b.float())


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # ratio = np.sqrt(1. - lam)
    cut_w = np.int(W * lam)
    cut_h = np.int(H * lam)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2