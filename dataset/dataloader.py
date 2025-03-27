from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import Sampler
import torch
import random
import pandas as pd

from util import TwoStreamBatchSampler, balance_data, balance_data_list
from dataset.Papilledema import papilledema_split, Papilledema  
import dataset.Papilledema


class MyRandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DataManager:
    
    def __init__(self, args, clean_indexs=None, pseudo_one_hot=None):
        
        self.args = args
        
        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)

        if args.dataset == 'Papilledema':
            train_dataset, val_dataset, test_dataset = self._get_dataset_papiledema(args, clean_indexs, pseudo_one_hot)
            self.num_class = dataset.Papilledema.N_CLASSES
            
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        val_num, test_num = len(val_dataset), len(test_dataset)
        
        self.labeled_idxs = list(self.label_idx)
        self.unlabeled_idxs = list(self.unlabel_idx)

        label_class_count = np.unique(self.label_data, return_counts=True)[1]
        ulabel_class_count = np.unique(self.unlabel_data, return_counts=True)[1]
        print("train_label_class_count: ", label_class_count, "train_ulabel_class_count:", ulabel_class_count)
        
        print("using {} train_lb_images, {} train_ulb_images, {} val_images, {} test_image".format(len(self.labeled_idxs), len(self.unlabeled_idxs), val_num, test_num))
        
        self.train_dataset = train_dataset
        
        batch_sampler = TwoStreamBatchSampler(self.labeled_idxs, self.unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
        
        self.train_dl = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_dl = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
        self.test_dl = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8, pin_memory=False)
    
    @staticmethod
    def __get_dataloader__(data, batch_size, num_workers, num_iters=1, train=True):
        
        if not train:
            return DataLoader(
                data,
                batch_size=batch_size,
                num_workers=num_workers
            )
        
        sampler = RandomSampler(data, replacement=True, num_samples=num_iters * batch_size, generator=None) 
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
        return DataLoader(
            data,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
        )

    def _get_dataset_papiledema(self, args, clean_indexs=None, pseudo_one_hot=None):
        
        train_path, val_path, test_path, train_targets, val_targets, test_targets = papilledema_split(args)
        idx, num_class_per = [], []
        classes = np.unique(train_targets)
        for t in range(len(classes)):
            idx.append(np.where(np.array(train_targets)==t)[0])
        for itm in idx:
            num_class_per.append(len(itm))
        self.num_class_per = num_class_per
        
        imgs_per_class = args.l_num // len(classes)
        # imgs_per_class = int(len(train_path) * args.l_ratio // len(classes))
        label_train_path, label_train_targets = list(), list()
        unlabel_train_path, unlabel_train_targets = list(), list()
        
        for cls in classes:
        
            img_idxs = np.where(train_targets == cls)[0]
            labeled_idx = np.random.choice(img_idxs, imgs_per_class, False)
        
            label_train_path.extend([train_path[i] for i in labeled_idx])
            label_train_targets.extend([train_targets[i] for i in labeled_idx])

            unlabeled_idx = np.setdiff1d(img_idxs, labeled_idx)    
            unlabel_train_path.extend([train_path[i] for i in unlabeled_idx])
            unlabel_train_targets.extend([train_targets[i] for i in unlabeled_idx])
            
        train_path = label_train_path + unlabel_train_path
        train_targets = label_train_targets + unlabel_train_targets

        self.label_idx = range(len(label_train_path))
        self.unlabel_idx = range(len(label_train_path), len(train_path))
        
        self.label_data = label_train_targets
        self.unlabel_data = unlabel_train_targets
        
        train_dataset = Papilledema(args, train_path, train_targets, train=True)
        # label_train_dataset = Papilledema(args, label_train_path, label_train_targets, train=True)
        # ulabel_train_dataset = Papilledema(args, unlabel_train_path, unlabel_train_targets, train=True)
        # val_train_dataset = Papilledema(args, train_path, train_targets)
        val_dataset = Papilledema(args, val_path, val_targets)
        test_dataset = Papilledema(args, test_path, test_targets)
        
        if clean_indexs is not None:
            clean_indexs = [int(idx) for idx in clean_indexs]
            clean_path = [train_path[i] for i in clean_indexs]
            clean_targets = torch.argmax(torch.tensor(pseudo_one_hot), dim=1).tolist()
            clean_path = label_train_path + clean_path
            clean_targets = label_train_targets + clean_targets
        
            clean_dataset = Papilledema(args, clean_path, clean_targets, train=True)
            # batch_sampler = TwoStreamBatchSampler(list(self.label_idx), list(range(len(label_train_path), len(clean_path))), args.batch_size, args.batch_size-args.labeled_bs)
            # self.clean_loader = DataLoader(clean_dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)        
            self.clean_loader = self.__get_dataloader__(clean_dataset, args.batch_size, num_workers=8, num_iters=args.num_train_iters)  

        return train_dataset, val_dataset, test_dataset


