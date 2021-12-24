import os
import shutil
import torch
import logging
from collections import OrderedDict
from sklearn import metrics
import math
import numpy as np
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from .config import *


def accuracy(output, target, num_classes):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        acc = []
        mean_acc = 0
        ones_m = torch.ones(output.size()).cuda()
        zeros_m = torch.zeros(output.size()).cuda()

        score_map = torch.where(output>cls_thresh, ones_m, zeros_m)

        correct = score_map.eq(target).type(torch.FloatTensor)
        batch_size = float(output.size(0))
        
        for i in range(num_classes):
            tmp = float(correct[:,i].sum())/batch_size
            acc.append(tmp)
            mean_acc += tmp
        mean_acc = float(mean_acc) / float(num_classes)

        return mean_acc, acc

def print_result(display_str, result_class, classes):
    num_classes = len(classes)
    display_str = display_str
    for i in range(num_classes):
        if i < num_classes-1:
            display_str += '{} {:.4f}, '.format(classes[i], result_class[i])
        else:
            display_str += '{} {:.4f}'.format(classes[i], result_class[i])
    logging.info(display_str)

def print_thresh_result(display_str, result, thresh, classes):
    display_str = display_str
    for idx in range(len(classes)):
        display_str += classes[idx] + '{'
        for i in range(len(thresh)):
            if i+1 != len(thresh):
                display_str += '{:.4f}({}), '.format(result[idx][i], thresh[i])
            else:
                display_str += '{:.4f}({})'.format(result[idx][i], thresh[i])
        if idx+1 != len(classes):
            display_str += '}; '
        else:
            display_str += '}'
    logging.info(display_str)


def calculate_auc(y_pred, y_gt, num_classes):
    '''calculate the mean AUC'''
    auc_each_class = []
    nan_index = []

    mean_auc = 0

    for index in range(num_classes):

        pred = y_pred[:, index]
        label = y_gt[:, index]

        fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)

        auc = metrics.auc(fpr, tpr)

        if(math.isnan(auc)):
            nan_index.append(index)
            auc = 0.0

        auc_each_class.append(auc)
        mean_auc += auc

    mean_auc = float(mean_auc) / float(num_classes)

    return mean_auc, auc_each_class

def IOU(pred, target, n_classes):
    ious = []
    mean_iou = 0
    for idx in range(n_classes): 
        pred_class = pred[:, idx, :, :].contiguous().view(-1)
        target_class = target[:, idx, :, :].contiguous().view(-1)
        pred_inds = pred_class == 1
        target_inds = target_class == 1
        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        ious.append(float(intersection) / float(max(union, 1)))
    for i in range(len(ious)):
        mean_iou += ious[i]
    mean_iou = float(mean_iou) / float(n_classes)
    return np.array(ious), mean_iou

def print_iou(iou, Xray_CLASSES, str_head='IOU for All Classes:'):
    num_classes = len(Xray_CLASSES)
    display_str = str_head + ' '
    for i in range(num_classes):
        if i < num_classes-1:
            display_str += '{} {:.4f}, '.format(Xray_CLASSES[i], iou[i])
        else:
            display_str += '{} {:.4f}'.format(Xray_CLASSES[i], iou[i])
    return display_str


def save_checkpoint(net, arch, epoch, _bestauc=False, _bestiou=False, _bestior=False, best=0):
    savepath = os.path.join(model_savepath, arch)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file_name = os.path.join(savepath, "{}_epoch_{:0>4}".format(arch, epoch)+ '.pth')
    torch.save(net.state_dict(), file_name)
    remove_flag = False
    if _bestiou:
        best_name = os.path.join(savepath, "{}_best_tiou".format(arch)+ '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_tiou".format(arch)+ '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best mean tiou: {}'.format(best)+'\n')
        file.close()
    if _bestior:
        best_name = os.path.join(savepath, "{}_best_tior".format(arch)+ '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_tior".format(arch)+ '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best mean tior: {}'.format(best)+'\n')
        file.close()
    if _bestauc:
        best_name = os.path.join(savepath, "{}_best_auc".format(arch)+ '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_auc".format(arch)+ '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best auc: {}'.format(best)+'\n')
        file.close()
    if remove_flag:
        os.remove(file_name)
        

def load_checkpoint(net, model_path, _sgpu=True):
    state_dict = torch.load(model_path)
    if _sgpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head != 'module.':
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    logging.info('Load resume network...')

def set_requires_grad(net, fixed_layer, _sgpu=True):
    update_flag = {}
    for name, _ in net.named_parameters():
        # print(name)
        update_flag[name] = 0
        for item in fixed_layer:
            if _sgpu:
                if name[:len(item)] == item:
                    # print('hehe')
                    update_flag[name] = 1
            else:
                if name[7:7+len(item)] == item:
                    # print('hehe')
                    update_flag[name] = 1

    for name, param in net.named_parameters():
        # print(name)
        if update_flag[name] == 1:
            param.requires_grad = False
        else:
            param.requires_grad = True

def adjust_learning_rate(optimizer, epoch, epoch_num, initial_lr, reduce_epoch, decay=0.1):
    epoch -= 1
    if reduce_epoch == 'dynamic':
        lr = initial_lr * (1 - math.pow(float(epoch)/float(epoch_num), power))
    else:
        lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_requires_grad(self, nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_flag=True):
        if sum_flag:
            self.val = val
            self.sum += val * n
        else:
            self.val = val / n
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count