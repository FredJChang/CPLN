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

N_CLASSES = 2
N_CLASSES_TYPE = ['benign', 'malignant']

class_to_numeric = {
    'benign': 0,
    'malignant': 1
}

def BreastMR_split(args):
    base_path = args.data_root + args.dataset
    data = []
    for c in ['benign', 'malignant']:
        class_path = base_path + "/" + c + "/"
        for people in os.listdir(class_path):
            people_path = os.path.join(class_path, people)
            for people_single in os.listdir(people_path):
                numeric_class = class_to_numeric[c]
                data_class = [(os.path.join(people_path, people_single), numeric_class)]
                data.extend(data_class)

    if args.mode == 'KFold':
        kfold = KFold(5, shuffle = True, random_state = args.seed)
        splits = list(kfold.split(data))
        train_index, val_index = splits[args.fold]

        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]

        train_path, train_targets = [item[0] for item in train_data], [item[1] for item in train_data]
        val_path, val_targets = [item[0] for item in val_data], [item[1] for item in val_data]
        test_path, test_targets = val_path, val_targets

    else:
        image, label = zip(*data)
        train_path, rest_data_path, train_targets, rest_targets = train_test_split(image, label, 
                                                                            test_size=1-args.train_ratio, 
                                                                            random_state=args.seed, stratify=label)

        val_path, test_path, val_targets, test_targets = train_test_split(rest_data_path, rest_targets, 
                                                                        test_size=args.test_ratio/(args.val_ratio+args.test_ratio), 
                                                                        random_state=args.seed, stratify=rest_targets)

    return train_path, val_path, test_path, train_targets, val_targets, test_targets


class BreastMR(Dataset):

    def __init__(self, args, image, targets, clean_targets=None, train=False, clean=False):

        super(Dataset, self).__init__()
        self.clean = clean
        self.train = train
        self.image = image
        self.targets = targets
        self.clean_targets = clean_targets
        self.n_classes = N_CLASSES
        self.n_class_type = N_CLASSES_TYPE

        self.weak_transform = transforms.Compose([
                                              transforms.Resize((128, 128)),
                                            #   transforms.RandomHorizontalFlip(),
                                            #   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                                            #   transforms.RandomVerticalFlip(),
                                            #   transforms.RandomRotation(10),
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
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image.resize((128, 128)))
        
        label = self.targets[index]
        label = F.one_hot(torch.tensor(label), num_classes = self.n_classes)
        study = image_path[image_path.rfind('/') + 1:]
        
        strong_image = self.strong_transform(image)
        # strong_image = cutout(strong_image)

        
        if self.clean == True:
            clean_targets = self.clean_targets[index]
            
            return dict(index=index, image_array = image_array, label = label, study = study, clean_targets = clean_targets,
                base_image = self.base_transform(image), weak_image = self.weak_transform(image), strong_image = strong_image)

        
        if self.train == True:
            return dict(index=index, image_array = image_array, label = label, study = study, 
                        base_image = self.base_transform(image), weak_image = self.weak_transform(image), strong_image = strong_image)
                        

        return dict(index=index, image_array = image_array,
                    base_image = self.base_transform(image),
                    label = label,
                    study = study)
        
    def __len__(self) -> int:
        
        return len(self.image)

    
