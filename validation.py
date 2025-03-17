import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.metrics import compute_metrics
from utils.util import AverageMeter
CE = torch.nn.CrossEntropyLoss(reduction='mean')

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def epochVal_metrics(model, dataLoader, mode='None'):
    training = model.training
    model.eval()
    
    losses = AverageMeter()
    # meters = MetricLogger()
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():

        
        for i, batch in enumerate(dataLoader):
            study, image, label = batch['study'], batch['base_image'], batch['label']
            image, label = image.cuda(), label.cuda()
            
            # for fwd_pass in range(5):
            #     enable_dropout(model)
            #     _, output = model(image)
            #     output = F.softmax(output, dim=1)
            #     if fwd_pass == 0:
            #         outputs = output
            #     else:
            #         outputs += output

            # output = outputs / 5
            
            if mode == 'val':
                for fwd_pass in range(5):
                    enable_dropout(model)
                    _, output = model(image)
                    output = F.softmax(output, dim=1)
                    if fwd_pass == 0:
                        outputs = output
                    else:
                        outputs += output

                output = outputs / 5
            else:
                _, output = model(image)
                output = F.softmax(output, dim=1)
        
            loss = CE(output, label.float())
            
            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
            losses.update(loss.item())

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

    AUROCs, Accus, Senss, Specs, pre, Recall, F1 = compute_metrics(gt, pred, competition=True)
        
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    Pre_avg = np.array(pre).mean()
    Recall_avg = np.array(Recall).mean()
    F1_avg = np.array(F1).mean()
    print(f'{mode}: loss: {losses.avg:.5f} AUC-ROC: {AUROC_avg:.5f}  Accuracy: {Accus_avg:.5f}  Sensitivity: {Senss_avg:.5f}  Specificity: {Specs_avg:.5f}  Precision: {Pre_avg:.5f}  Recall: {Recall_avg:.5f}  F1 Score: {F1_avg:.5f}\n')
    
    model.train(training)

    return losses.avg, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, Pre_avg, Recall_avg, F1_avg



def epochVal_metrics_two(model1, model2, dataLoader, mode='None'):
    training1 = model1.training
    training2 = model2.training
    model1.eval()
    model2.eval()
    
    losses = AverageMeter()
    # meters = MetricLogger()
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():

        for i, batch in enumerate(dataLoader):
            study, image, label = batch['study'], batch['base_image'], batch['label']
            image, label = image.cuda(), label.cuda()
            
            _, output1 = model1(image)
            _, output2 = model2(image)
            output = (output1 + output2) / 2
            output = F.softmax(output, dim=1)
        
            loss = CE(output, label.float())
            
            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
            losses.update(loss.item())

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

    AUROCs, Accus, Senss, Specs, pre, Recall, F1 = compute_metrics(gt, pred, competition=True)
        
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    Pre_avg = np.array(pre).mean()
    Recall_avg = np.array(Recall).mean()
    F1_avg = np.array(F1).mean()
    print(f'{mode}: loss: {losses.avg:.5f} AUC-ROC: {AUROC_avg:.5f}  Accuracy: {Accus_avg:.5f}  Sensitivity: {Senss_avg:.5f}  Specificity: {Specs_avg:.5f}  Precision: {Pre_avg:.5f}  Recall: {Recall_avg:.5f}  F1 Score: {F1_avg:.5f}\n')
    
    model1.train(training1)
    model2.train(training2)

    return losses.avg, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, Pre_avg, Recall_avg, F1_avg
