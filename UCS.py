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
parser.add_argument('--warm_lr', type=float, default=0.0001, help='warm learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')

# parser.add_argument('--labeled_bs', type=int, default=16, help='training lb batch size')
parser.add_argument('--labeled_bs', type=int, default=8, help='training lb batch size')



parser.add_argument('--l_ratio', default=0.05, type=float, help="ratio for label/unlabel split")
parser.add_argument('--l_num', default=20, type=int, help="num for label data")

parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='beta')

parser.add_argument('--most_common_threshold', type=int, default=0.9)
parser.add_argument('--filter_ratio', type=float, default=0.3, help='filter_ratio')

parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--not_balance', action='store_false', help='Perform data balance augmentation')
parser.add_argument('--mode', type=str, choices=['KFold','tr_val'], default='tr_val')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 2).')
parser.add_argument('--seed', default=1337, type=int)

parser.add_argument('--data_root', type=str, default='/nfs/data/')
parser.add_argument('--model', type=str, choices=['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'AlexNet', 
                                                  'GoogLeNet', 'DenseNet121', 'DenseNet161', 'DenseNet201', 'DenseNet', 'wrn28'], default='DenseNet121')

parser.add_argument('--train_ratio', default=0.7)
parser.add_argument('--val_ratio', default=0.1)
parser.add_argument('--test_ratio', default=0.2)

parser.add_argument('--num_train_iters', type=int, default=2000)
parser.add_argument('--num_eval_iters', type=int, default=100)
parser.add_argument('--warmup_iters', type=int, default=2000)

parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', choices=['U-MultiClass', 'U-Ones'], help='label type')

parser.add_argument('--dataset', type=str, choices=['Chest', 'Nerthus', 'BreastMR', 'RPE', 'Head', 'Papilledema', 'Shenzhen', 'Chestxray', 'ISIC',
                                                    'BCI', 'Knee', 'Messidor'], default='Papilledema')
parser.add_argument('--devices', default='2,3,5', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')

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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

args.batch_size = args.batch_size
labeled_bs = args.labeled_bs
warm_lr = args.warm_lr
lr = args.lr

WCE_mean = losses.weighted_cross_entropy_loss()
CE_mean = torch.nn.CrossEntropyLoss(reduction='mean')
CE = torch.nn.CrossEntropyLoss(reduction='none')
BCE = torch.nn.BCEWithLogitsLoss(reduction="mean")
MSE = torch.nn.MSELoss()
Cosine = torch.nn.CosineSimilarity(dim=1)
FocalLoss = losses.MultiClassFocalLossWithAlpha()
semiloss = SemiLoss()

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [args.lr] * args.train_pseudo_history_epoch
beta1_plan = [mom1] * args.train_pseudo_history_epoch
for i in range(args.epoch_decay_start, args.train_pseudo_history_epoch):
    alpha_plan[i] = float(args.train_pseudo_history_epoch - i) / (args.train_pseudo_history_epoch - args.epoch_decay_start) * args.lr
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch): 
    for param_group in optimizer.param_groups: 
        param_group['lr']=alpha_plan[epoch] 
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1 

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        

# Network definition
def create_model(num_classes, ema=False):
    model_mapping = {
    'DenseNet121': DenseNet121(out_size=num_classes, mode=args.label_uncertainty, pretrained=True, drop_rate=args.drop_rate),
    'ResNet34': ResNet34(num_classes=num_classes),
    }
    net = model_mapping[args.model]
    if len(args.devices.split(',')) > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
            ema_model = copy.deepcopy(model)
            return ema_model
    return model

def train_pseudo_history(args, train_loader, val_dataloader, test_dataloader, model, ema_model1, ema_model2, optimizer, save_pseudo_path, l_dl, u_dl):

    recorder1 = [[] for i in range(train_loader.dataset.__len__())]
    recorder2 = [[] for i in range(train_loader.dataset.__len__())]
    alpha = args.alpha
    beta = args.beta  
    best_loss = np.inf
    best = 0
    curr_iter = 0
    global lr
    
    iters = []
    new_correct = []
    old_correct = []
    easy_accuracies = []
    fix_accuracies = []
    easy_counts = []
    fix_counts = []
    
    class_means = torch.zeros(n_class)
    class_stds = torch.ones(n_class)
    class_counts = torch.zeros(n_class)
    
    model.train()

    start_time = time.time()

    bar = tqdm(l_dl)
    for (batch_lb, batch_ulb) in tqdm(zip(l_dl, u_dl)):
        img_lb_w, img_lb_s, label_lb = batch_lb['weak_image'].cuda(), batch_lb['strong_image'].cuda(), batch_lb['label'].cuda()
        index, img_ulb_w, img_ulb_s, label_ulb = batch_ulb['index'], batch_ulb['weak_image'].cuda(), batch_ulb['strong_image'].cuda(), batch_ulb['label'].cuda()
        
        l_image = torch.cat([img_lb_w, img_lb_s])
        l_target = torch.cat([label_lb, label_lb])
        u_image = torch.cat([img_ulb_w, img_ulb_s])
        u_target = label_ulb
        
        # 1. Calculate loss for labeled data
        feat_l, logits = model(l_image)
        ce = CE_mean(logits, l_target.float())
        
        # # 2. Calculate loss for unlabeled data
        feat_u, logits_u = model(u_image)
        logits_u_w, logits_u_s = logits_u.chunk(2)
        feat_u_w, feat_u_s = feat_u.chunk(2)
        
        # 3. Calculate pseudo labels and their corresponding loss values
        probs = torch.softmax(logits_u_w, dim=-1)
        max_probs, pseudo_labels = torch.max(probs, dim=-1)
        
        
        with torch.no_grad():
            feat_ema, logits_ema = ema_model1(img_ulb_w)
            probs_ema = torch.softmax(logits_ema, dim=-1)
            probs = (probs + probs_ema) / 2.0
            max_probs, pseudo_labels = torch.max(probs, dim=-1)
        
        
        strong_probs = torch.softmax(logits_u_s, dim=-1)
        max_probs_s, targets_u_s = torch.max(strong_probs, dim=-1)
        
        loss_values = F.cross_entropy(logits_u_s, pseudo_labels, reduction='none')
        
        # 4. Dynamic threshold based on loss and confidence
        model.eval()
        with torch.no_grad():
            for fwd_pass in range(5):
                enable_dropout(model)
                _, output = model(u_image)
                output = torch.softmax(output, dim=-1)
                output = output / 0.95
                output_w, ouput_s = output.chunk(2)
                loss_drop = F.cross_entropy(ouput_s, pseudo_labels, reduction='none')
                
                np_output = ouput_s.detach().cpu().numpy()[np.newaxis, :, :]
                np_loss = loss_drop.detach().cpu().numpy()[np.newaxis, :]
            
                if fwd_pass == 0:
                    dropout_pred = np_output
                    dropout_loss = np_loss
                else:
                    dropout_pred = np.vstack((dropout_pred, np_output))
                    dropout_loss = np.vstack((dropout_loss, np_loss))   

            epsilon = sys.float_info.min
            
            mean_pred = np.mean(dropout_pred, axis=0) + epsilon
            mean_loss = np.mean(dropout_loss, axis=0) + epsilon

            entropy = -np.sum(mean_pred * np.log(mean_pred), axis=-1)
            entropy[np.isnan(entropy)] = 0
            entropy = torch.tensor(entropy).cuda()
        
        model.train()
        
        loss_threshold = mean_loss.mean() +  mean_loss.std()
        confidence_threshold = mean_pred.mean() +  mean_pred.std()

        # 5. Mask for high confidence and low loss samples
        clean_mask = (max_probs_s > confidence_threshold) & (loss_values < loss_threshold) 

        
        noise_mask = ~clean_mask 
 
        if torch.any(hard_mask):
            mixed_image, y_a, y_b, lam= mixup_tru_unc(l_image, img_ulb_s[noise_mask], l_target, probs[noise_mask])
            feat, output_mix = model(mixed_image)
            mix_loss = mixup_criterion(output_mix, y_a, y_b, lam)
        else: 
            mix_loss = torch.tensor(0.0) 
        
        # 6. Weight calculation
        weights = 1.0 / (1.0 + entropy)
        
        # 9. Select 
        high_confidence_logits = logits_u_s[clean_mask]
        selected_pseudo_labels = pseudo_labels[clean_mask]
        selected_weights = weights[clean_mask]
        
        # 8. Compute weighted loss
        Lu = (selected_weights  * F.cross_entropy(high_confidence_logits, selected_pseudo_labels, reduction='none')).mean()

        loss = ce + alpha * Lu + beta * mix_loss


        # 10. Update EMA model
        with torch.no_grad():
            activations, logits_ema1 = ema_model1(img_ulb_w)
            activations, logits_ema2 = ema_model2(img_ulb_s)
            pred1 = F.softmax(logits_ema1, dim=1).cpu().data
            pred2 = F.softmax(logits_ema2, dim=1).cpu().data
            
            for i, ind in enumerate(index + label_num):
                current_value1 = np.array(pred1[i].numpy().tolist())
                current_value2 = np.array(pred2[i].numpy().tolist())
                recorder1[ind].append(current_value1.tolist())
                recorder2[ind].append(current_value2.tolist())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (curr_iter + 1) % args.num_eval_iters == 0:        
            val_loss, AUROCs, Accus, Senss, Specs, Pre, Recall, F1 = epochVal_metrics(model, val_dataloader, mode='val')
            test_loss, test_AUROCs, test_Accus, test_Senss, test_Specs, test_Pre, test_Recall, test_F1 = epochVal_metrics(model, test_dataloader, mode='test') 
            
        update_ema_variables(model, ema_model1, 0.9, curr_iter)
        update_ema_variables(model, ema_model2, 0.9, curr_iter)


    with open("%s/record1.p"%(args.save_folder), 'wb') as recordf1:
        pickle.dump(recorder1, recordf1)
    with open("%s/record2.p"%(args.save_folder), 'wb') as recordf2:
        pickle.dump(recorder2, recordf2)  
    
    return recorder1, recorder2
