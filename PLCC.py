# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--warmup_epoch', type=int,  default=30, help='warmup epoch')
parser.add_argument('--train_pseudo_history_epoch', type=int, default=50)
parser.add_argument('--global_epoch', type=int, default=10, help='maximum epoch number to train')
parser.add_argument('--epoch_decay_start', type=int, default=10, help='epoch from which to start lr decay')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')

parser.add_argument('--labeled_bs', type=int, default=8, help='training lb batch size')
parser.add_argument('--l_num', default=20, type=int, help="num for label data")

parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='beta')

parser.add_argument('--most_common_threshold', type=int, default=0.9)
parser.add_argument('--filter_ratio', type=float, default=0.3, help='filter_ratio')

parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--not_balance', action='store_false', help='Perform data balance augmentation')
parser.add_argument('--mode', type=str, choices=['KFold','tr_val'], default='tr_val')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 2).')
parser.add_argument('--seed', default=None, type=int)

parser.add_argument('--data_root', type=str, default='/nfs/data/')
parser.add_argument('--model', type=str, choices=['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'AlexNet', 
                                                  'GoogLeNet', 'DenseNet121', 'DenseNet161', 'DenseNet201', 'DenseNet', 'wrn28'], default='DenseNet121')

parser.add_argument('--train_ratio', default=0.7)
parser.add_argument('--val_ratio', default=0.1)
parser.add_argument('--test_ratio', default=0.2)

parser.add_argument('--num_train_iters', type=int, default=2000)
parser.add_argument('--num_eval_iters', type=int, default=100)

parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', choices=['U-MultiClass', 'U-Ones'], help='label type')

parser.add_argument('--dataset', type=str, choices=['Chest', 'Nerthus', 'BreastMR', 'RPE', 'Head', 'Papilledema', 'Shenzhen', 'Chestxray', 'ISIC',
                                                    'BCI', 'Knee', 'Messidor'], default='BreastMR')
parser.add_argument('--devices', default='0,1,2', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

import pathlib
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib import tzip
from pathlib import Path
import numpy as np
import copy
import time
from PIL import Image
import pickle
from collections import Counter
import sys
import pandas as pd

import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset, Dataset, Subset, ConcatDataset

from networks import AlexNet, GoogLeNet, ResNet10, ResNet18, ResNet34, ResNet50, ResNet101, MetaNet, wrn_28_2, wrn_28_8
from networks.get_densenet import DenseNet121, DenseNet161, DenseNet201
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from torchvision import transforms
import itertools

from dataset.dataloader import DataManager
from dataset.utils import modify_loader

from utils import losses
from utils.losses import contrastive_loss, simclr_loss, relation_mse_loss, SemiLoss
from utils.metrics import save_args, generate_result, loss_curve, accuracy
from utils.tsne import tsne_2d, tsne_3d, tsne_3d_echarts
from utils.util import AverageMeter, mixup_criterion, mixup_data, most_common_value_with_threshold, mixup_tru_unc
from validation import epochVal_metrics
from utils.metrics import compute_metrics

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.cluster import KMeans
import subprocess
import json

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

args.batch_size = args.batch_size
labeled_bs = args.labeled_bs
lr = args.lr


def history_clean(args, recorder1, recorder2, train_loader, model, label_num, datamaker):
    
    '''add all avg logits to dataset'''
    recorder1_array = np.array([np.array(logits) for logits in recorder1], dtype=object)
    recorder2_array = np.array([np.array(logits) for logits in recorder2], dtype=object)
    logits_combined = 0.5 * np.array(recorder1_array) + 0.5 * np.array(recorder2_array) # only array can add
    
    logits_combined_avg_list = []
    for item in logits_combined:
        if np.isnan(item).all():
            avg_item = np.pad(item, (0, n_class), constant_values=-1) # pad -1 to nan
        else:
            avg_item = np.mean(item, axis=0).tolist() # avg logits
        logits_combined_avg_list.append(avg_item)
    logits_combined_avg_array = np.vstack(logits_combined_avg_list)
    clean_indices = list(set(range(len(datamaker.train_dataset)))-set(range(label_num)))

    error_rate(args, model, datamaker, logits_combined_avg_array, clean_indices, label_num)
    
    # clean_indices = []
    # ''' 
    # idea1、SHCS
    # '''
    pseudo_label = {}
    labels_recorder1 = [[np.argmax(prob_dist) for prob_dist in record] for record in recorder1]
    labels_recorder2 = [[np.argmax(prob_dist) for prob_dist in record] for record in recorder2]
    for index in range(len(labels_recorder1)):
        if index > label_num : # except label data
            record_recorder1, record_recorder2 = labels_recorder1[index], labels_recorder2[index]
            if len(record_recorder1) != 0 and len(record_recorder2) != 0: 
                # unlabel data 
                v1 = most_common_value_with_threshold(record_recorder1, args.most_common_threshold)
                v2 = most_common_value_with_threshold(record_recorder2, args.most_common_threshold)
                if v1 is not None and v2 is not None and v1 == v2:
                    pseudo_label[index] = Counter(record_recorder1).most_common(1)[0][0]
                    clean_indices.append(index)
                # if record_recorder1[:] == record_recorder2[:]:
                #     if all(x == record_recorder1[-1] for x in record_recorder1[:]) and all(x == record_recorder2[-1] for x in record_recorder2[:]):
                #         pseudo_label[index] = record_recorder1[-1]
    
    error_rate(args, model, datamaker, logits_combined_avg_array, clean_indices, label_num)
                    
                               
    '''
    idea2、FHCS
    '''
    selected_data_logits1 = {}
    for pseudo_index, pseudo_value in pseudo_label.items():
        selected_data_logits1[pseudo_index] = [row[pseudo_value] for row in recorder1[pseudo_index]]
    average_logits_per_data1 = {key: np.mean(logits, axis=0) for key, logits in selected_data_logits1.items()}
    index1 = np.argsort(list(average_logits_per_data1.values()))
    filter_ratio = args.filter_ratio
    indices1 = index1[int(len(index1) * filter_ratio):]
    clean_indices1 = [list(average_logits_per_data1.keys())[i] for i in indices1]
    
    selected_data_logits2 = {}
    for pseudo_index, pseudo_value in pseudo_label.items():
        selected_data_logits2[pseudo_index] = [row[pseudo_value] for row in recorder2[pseudo_index]]
    average_logits_per_data2 = {key: np.mean(logits, axis=0) for key, logits in selected_data_logits2.items()}
    index = np.argsort(list(average_logits_per_data2.values()))
    filter_ratio = args.filter_ratio
    indices2 = index[int(len(index) * filter_ratio):]
    clean_indices2 = [list(average_logits_per_data2.keys())[i] for i in indices2]
    clean_indices = list(set(clean_indices1) & set(clean_indices2) & set(clean_indices))

    error_rate(args, model, datamaker, logits_combined_avg_array, clean_indices, label_num)



    clean_index = clean_indices #+ list(range(label_num)) # add label data
    dataset = datamaker.train_dataset
    real_indices = list(range(len(dataset)))
    
    clean_set = Subset(dataset, clean_index)
    
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, num_workers=8, pin_memory=True)
     
    

    # model.eval()
    # losses = torch.zeros(len(train_loader.dataset))   
    # with torch.no_grad():
    #     total_loss = 0
    #     indexs = []
    #     for i, batch in enumerate(tqdm(clean_loader)):
    #         index, image, label = batch['index'], batch['base_image'].cuda(), batch['label'].cuda()
    #         pseudo = torch.from_numpy(logits_combined_avg_array[index]).cuda()
    #         if pseudo.dim() == 1:
    #             pseudo = pseudo.unsqueeze(0) 
    #         label = torch.argmax(label, dim=1)
    #         _, outputs = model(image)
    #         loss = F.cross_entropy(outputs, pseudo, reduction='none')
    #         total_loss += loss.cpu().detach().numpy().mean()
            
    #         for b in range(image.size(0)):
    #             losses[index[b]]=loss[b]
    #             indexs.append(index[b])


    #     losses = losses[indexs]
    #     losses = (losses - losses.min()) / (losses.max() - losses.min())
    #     input_loss = losses.reshape(-1, 1)
    #     gmm = gmm_fit(model, train_loader)


    #     prob = gmm.predict_proba(input_loss)
    #     prob = prob[:,gmm.means_.argmin()]  
    #     t = np.percentile(prob, 10)
    #     pred = (prob > 0.4) 
    #     indexs = np.array([idx.item() for idx in indexs])  # tensor to numpy
    #     gmm_index = indexs[pred].tolist()
        
    #     clean_index = gmm_index
    
    clean_logits = logits_combined_avg_array[clean_index]
    pseudo = np.argmax(clean_logits, axis=1)
    pseudo_one_hot = np.eye(n_class)[pseudo]
    
    clean_loader = DataManager(args, clean_index, pseudo_one_hot).clean_loader # add label data in Datamanager
    
    # after gmm ratio
    # error_rate(args, model, datamaker, logits_combined_avg_array, clean_index, label_num) 


def main(args):
    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.exp_name = f"{current_experiment_time}_{args.dataset}_{args.l_num}_{args.seed}_final" \
    
    args.save_folder = pathlib.Path(f"/data-hhd/result/final/new/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    save_args(args)
    
    save_model_path = os.path.join(args.save_folder, 'save_models/')
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    
    datamaker = DataManager(args)

    label_train_loader = datamaker.train_lb_dl
    train_loader = datamaker.train_dl
    ulabel_train_loader = datamaker.train_ulb_dl
    val_dataloader = datamaker.val_dl
    test_dataloader = datamaker.test_dl
    
    
    global label_num
    global n_class
    global num_class_per
    label_num = len(datamaker.labeled_idxs)
    
    n_class = datamaker.num_class
    num_class_per = datamaker.num_class_per

    model = create_model(n_class)
    model1 = create_model(n_class)

    clean_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=5e-4)
    
    model.train()

    state = torch.load(model_path)
    model.load_state_dict(state)
    test_loss, AUROCs, Accus, Senss, Specs, Pre, Recall, F1 = epochVal_metrics(model, test_dataloader, mode='test')
    recorder1 = pickle.load(open('/data-hhd/result/test/{}_{}_{}/record1.p'.format(args.dataset, args.l_num, args.seed), 'rb'))
    recorder2 = pickle.load(open('/data-hhd/result/test/{}_{}_{}/record2.p'.format(args.dataset, args.l_num, args.seed), 'rb'))
    train_loader = history_clean(args, recorder1, recorder2, train_loader, model, label_num, datamaker)
    train_pseudo(args, train_loader, val_dataloader, test_dataloader, model, clean_optimizer, model1)

