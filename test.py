import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')

parser.add_argument('--labeled_bs', type=int, default=8, help='training lb batch size')

parser.add_argument('--not_balance', action='store_false', help='Perform data balance augmentation')
parser.add_argument('--mode', type=str, choices=['KFold','tr_val'], default='tr_val')
parser.add_argument('--devices', default='0,1,2', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 2).')
parser.add_argument('--seed', default=123, type=int)

parser.add_argument('--dataset', type=str, choices=['Chestxray', 'Nerthus', 'BreastMR', 'Papilledema'], default='Papilledema')
parser.add_argument('--data_root', type=str, default='/nfs/data/')
parser.add_argument('--model', type=str, choices=['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'AlexNet', 
                                                  'GoogLeNet', 'DenseNet121', 'DenseNet161', 'DenseNet201', 'DenseNet', 'wrn28'], default='DenseNet121')
parser.add_argument('--train_ratio', default=0.7)
parser.add_argument('--val_ratio', default=0.1)
parser.add_argument('--test_ratio', default=0.2)

parser.add_argument('--l_num', default=15, type=int, help="num for label data")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices


import numpy as np
import copy
import torch
import torch.nn.functional as F
from getdensenet import DenseNet121, DenseNet161, DenseNet201
from dataset.dataloader import DataManager
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from imblearn.metrics import sensitivity_score, specificity_score


def compute_metrics(gt, pred, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    AUROCs, Accus, Senss, Specs, Pre, Recall, F1 = [], [], [], [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    THRESH = 0.18

    num_class = gt.size(1)
    indexes = range(num_class)
    
    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)
        
        try:
            Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Senss.append(sensitivity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing precision for {}.'.format(i))
            Senss.append(0)
        
        try:
            Specs.append(specificity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Specs.append(0)

        try:
            Pre.append(precision_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Pre.append(0)
    
        try:
            F1.append(f1_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            F1.append(0)
            
        try:
            Recall.append(recall_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Recall.append(0)
            
    
    return AUROCs, Accus, Senss, Specs, Pre, Recall, F1

def epochVal_metrics(model, dataLoader, mode='None'):
    training = model.training
    model.eval()
    
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
            
            _, output = model(image)
            output = F.softmax(output, dim=1)
                    
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

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

    AUROCs, Accus, Senss, Specs, pre, Recall, F1 = compute_metrics(gt, pred, competition=True)
        
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    Pre_avg = np.array(pre).mean()
    # Recall_avg = np.array(Recall).mean()
    F1_avg = np.array(F1).mean()
    print(f'{mode}:  AUC-ROC: {AUROC_avg:.5f}  Accuracy: {Accus_avg:.5f}  Sensitivity: {Senss_avg:.5f}  Specificity: {Specs_avg:.5f}  Precision: {Pre_avg:.5f}  F1 Score: {F1_avg:.5f}\n')
    
    model.train(training)

    return AUROC_avg, Accus_avg, Senss_avg, Specs_avg, Pre_avg, F1_avg

# Network definition
def create_model(num_classes, ema=False):
    model_mapping = {
    'DenseNet121': DenseNet121(out_size=num_classes, mode='U-Ones', pretrained=True, drop_rate=0.2),
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


datamaker = DataManager(args)
test_dataloader = datamaker.test_dl
model = create_model(3)
state = torch.load('SSL/test/Papilledema15.pth') 
model.load_state_dict(state)
AUROCs, Accus, Senss, Specs, Pre, F1 = epochVal_metrics(model, test_dataloader, mode='test')