import os
import sys
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import torch
from .utils import noisify, RandAugmentMC, cutout


sys.path.append('..')

N_CLASSES = 4
N_CLASSES_TYPE = ['normal', 'adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma']

class_to_numeric = {
    'normal': 0,
    'adenocarcinoma': 1,
    'large.cell.carcinoma': 2,
    'squamous.cell.carcinoma':3  
}

def chest_split(args):
    train_path, val_path, test_path = [], [], []
    train_targets, val_targets, test_targets = [], [], []
    base_path = args.data_root + args.dataset

    for a in ['train', 'val', 'test']:   
        for c in ['normal', 'adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma']:
            class_path = base_path + "/" + a + "/" + c + "/"
            for subfiles in os.listdir(class_path):
                    data_path = os.path.join(class_path, subfiles)
                    numeric_class = class_to_numeric[c]
                    # Split data into train, val, test
                    if a == 'train':
                        train_path.append(data_path)
                        train_targets.append(numeric_class)
                    elif a == 'val':
                        val_path.append(data_path)
                        val_targets.append(numeric_class)
                    else:
                        test_path.append(data_path)
                        test_targets.append(numeric_class)

    return train_path, val_path, test_path, train_targets, val_targets, test_targets


class Chest(Dataset):

    def __init__(self, args, image, targets, train=False, eval_train=False):

        super(Dataset, self).__init__()
        self.eval_train = eval_train
        self.train = train
        self.image = image
        self.targets = targets
        self.n_classes = N_CLASSES
        self.n_class_type = N_CLASSES_TYPE

        self.weak_transform = transforms.Compose([
                                              transforms.Resize((128, 128)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.327, 0.327, 0.327],
                                                                   [0.196, 0.196, 0.196])
                                              ])
        self.base_transform = transforms.Compose([
                                              transforms.Resize((128, 128)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.327, 0.327, 0.327],
                                                                   [0.196, 0.196, 0.196])
                                          ])
        
        self.strong_transform = transforms.Compose([
                                            transforms.Resize((128, 128)),
                                            transforms.RandomHorizontalFlip(),   
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.327, 0.327, 0.327],
                                                                [0.196, 0.196, 0.196])
                                            ])



    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_path = self.image[index].strip()
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_array = np.array(image.resize((128, 128)))
        
        label = self.targets[index]
        label = F.one_hot(torch.tensor(label), num_classes = self.n_classes)
        study = image_path
        
        weak_image = self.weak_transform(image)
        # weak_image = cutout(weak_image)
        
        if self.train == True:
            return dict(index=index, image_array = image_array, label = label, study = study, 
                        base_image = self.base_transform(image), weak_image = weak_image, strong_image = self.strong_transform(image))
                        

        return dict(index=index, image_array = image_array,
                    base_image = self.base_transform(image),
                    label = label,
                    study = study)
        
    def __len__(self) -> int:
        
        return len(self.image)

    