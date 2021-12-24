import os
import gc
import time
import glob
import json
import cv2
import matplotlib.cm as cm
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from .tools import *
from .config import *
from . import vis_util
from . import html
from data import joint_transforms
from data.custom_dataset import NIHDataset

cls_criterion = nn.BCELoss()
ex_criterion = nn.MSELoss(reduction='none')


def prepare_net(config, model, _use='train'):
    normalize = transforms.Normalize(mean=config['Means'], std=config['Stds'])
    if _use == 'train':
        if config['optim'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), 
                                        config['lr'], weight_decay=config['weight_decay'])
        if config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), 
                                        config['lr'], weight_decay=config['weight_decay'])
        elif config['optim'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), 
                                    config['lr'], momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
    
        train_joint_transformer = transforms.Compose([
            joint_transforms.JointResize(config['img_size'], config['resize_factor']),
            joint_transforms.JointRandomCrop(config['img_size']),
            joint_transforms.JointRandomHorizontalFlip()
            ])
        
        train_dataset = NIHDataset('train', config['DataRoot'], config['Data_CLASSES'], 
              config['trainSet'], helper=config['helperSet'], using_crf=config['Using_CRF'],
              joint_transform=train_joint_transformer,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  normalize,
            ]))
        config['train_length'] = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batchsize'], shuffle=True,
             num_workers=config['num_workers'])

        val_joint_transformer = transforms.Compose([
            joint_transforms.JointResize(config['img_size']),
            ])
        val_dataset_cls = NIHDataset('val-cls', config['DataRoot'], config['Data_CLASSES'], 
              config['valClsSet'], helper=None, using_crf=config['Using_CRF'],
              joint_transform=val_joint_transformer,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  normalize,
            ]))
        val_dataset_iou = NIHDataset('val-iou', config['DataRoot'], config['Data_CLASSES'], 
              config['valIoUSet'], helper=None, using_crf=config['Using_CRF'],
              joint_transform=val_joint_transformer,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  normalize,
            ]))
        config['val_length_cls'] = len(val_dataset_cls)
        config['val_length_iou'] = len(val_dataset_iou)
        val_loader_cls = torch.utils.data.DataLoader(
            val_dataset_cls, batch_size=config['batchsize'], shuffle=False,
            num_workers=config['num_workers'])
        val_loader_iou = torch.utils.data.DataLoader(
            val_dataset_iou, batch_size=1, shuffle=False,
            num_workers=config['num_workers'])

        val_loader = [val_loader_cls, val_loader_iou]

        return optimizer, train_loader, val_loader

    elif _use == 'test':
        val_joint_transformer = transforms.Compose([
            joint_transforms.JointResize(config['img_size']),
            ])
        val_dataset_cls = NIHDataset('val-cls', config['DataRoot'], config['Data_CLASSES'], 
              config['testClsSet'], helper=None, using_crf=config['Using_CRF'],
              joint_transform=val_joint_transformer,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  normalize,
            ]))
        val_dataset_iou = NIHDataset('val-iou', config['DataRoot'], config['Data_CLASSES'], 
              config['testIoUSet'], helper=None, using_crf=config['Using_CRF'],
              joint_transform=val_joint_transformer,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  normalize,
            ]))
        config['val_length_cls'] = len(val_dataset_cls)
        config['val_length_iou'] = len(val_dataset_iou)
        val_loader_cls = torch.utils.data.DataLoader(
            val_dataset_cls, batch_size=config['batchsize'], shuffle=False,
            num_workers=config['num_workers'])
        val_loader_iou = torch.utils.data.DataLoader(
            val_dataset_iou, batch_size=1, shuffle=False,
            num_workers=config['num_workers'])

        test_loader = [val_loader_cls, val_loader_iou]

        return test_loader

    elif _use == 'visual':
        val_joint_transformer = transforms.Compose([
            joint_transforms.JointResize(config['img_size']),
            ])
        val_dataset_iou = NIHDataset('val-iou', config['DataRoot'], config['Data_CLASSES'], 
              config['testIoUSet'], helper=None, using_crf=config['Using_CRF'],
              joint_transform=val_joint_transformer,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  normalize,
            ]))
        config['val_length_iou'] = len(val_dataset_iou)
        visual_loader = torch.utils.data.DataLoader(
            val_dataset_iou, batch_size=1, shuffle=False,
            num_workers=config['num_workers'])

        return visual_loader


def train_net(visualizer, optimizer, train_loader, val_loader, model, config):
    val_loader_cls = val_loader[0]
    val_loader_iou = val_loader[1]
    best_auc = 0
    best_tiou = 0
    best_tior = 0
    
    if config['lr_decay'] == None:
        lr_decay = 0.1
    else:  
        lr_decay = config['lr_decay']
    for epoch in range(1, config['num_epoch']+1):
        adjust_learning_rate(optimizer, epoch, config['num_epoch'], config['lr'], 
                            config['lr_decay_freq'], lr_decay)

        train(visualizer, train_loader, model, optimizer, epoch, config)
        
        if epoch % config['test_freq'] == 0:
            TIOU, TIOR, auc_cls, all_auc_cls, acc_cls, all_acc_cls, auc_ano, acc_ano = val(val_loader_cls, val_loader_iou, model, config)
            logging.info('Test-Cls: Mean ACC:  {:.4f}, Mean AUC: {:.4f}'.format(acc_cls, auc_cls))
            print_result('Test-Cls: ACC for All Classes: ', all_acc_cls, config['Data_CLASSES'])
            print_result('Test-Cls: AUC for All Classes: ', all_auc_cls, config['Data_CLASSES'])
            logging.info('Test-Ano: ACC: {:.4f}, AUC: {:.4f}'.format(acc_ano, auc_ano))
            print_thresh_result('Test-TIoU: ', TIOU, thresh_TIOU, config['Data_CLASSES'])
            print_thresh_result('Test-TIoR: ', TIOR, thresh_TIOR, config['Data_CLASSES'])

            mTIOU = 0.
            len_TIOU = TIOU.shape[1]
            for idx in range(len(config['Data_CLASSES'])):
                mTIOU += TIOU[idx].sum()/float(len_TIOU)
            mTIOU /= float(len(config['Data_CLASSES']))

            mTIOR = 0.
            len_TIOR = TIOR.shape[1]
            for idx in range(len(config['Data_CLASSES'])):
                mTIOR += TIOR[idx].sum()/float(len_TIOR)
            mTIOR /= float(len(config['Data_CLASSES']))

            if auc_cls >= best_auc:
                save_checkpoint(model, config['arch'], epoch, _bestauc=True, best=auc_cls)
                best_auc = auc_cls
            if mTIOU >= best_tiou:
                save_checkpoint(model, config['arch'], epoch, _bestiou=True, best=mTIOU)
                best_tiou = mTIOU
            if mTIOR >= best_tior:
                save_checkpoint(model, config['arch'], epoch, _bestior=True, best=mTIOR)
                best_tior = mTIOR
        if epoch % config['save_model_freq'] == 0:
            save_checkpoint(model, config['arch'], epoch)
    save_checkpoint(model, config['arch'], epoch)


def test_net(test_loader, model, config, infer_iter):
    val_loader_cls = test_loader[0]
    val_loader_iou = test_loader[1]
    TIOU, TIOR, auc_cls, all_auc_cls, acc_cls, all_acc_cls, auc_ano, acc_ano = test(val_loader_cls, val_loader_iou, model, config, infer_iter)
    logging.info('Test-Cls: Mean ACC:  {:.4f}, Mean AUC: {:.4f}'.format(acc_cls, auc_cls))
    print_result('Test-Cls: ACC for All Classes: ', all_acc_cls, config['Data_CLASSES'])
    print_result('Test-Cls: AUC for All Classes: ', all_auc_cls, config['Data_CLASSES'])
    logging.info('Test-Ano: ACC: {:.4f}, AUC: {:.4f}'.format(acc_ano, auc_ano))
    print_thresh_result('Test-TIoU: ', TIOU, thresh_TIOU, config['Data_CLASSES'])
    print_thresh_result('Test-TIoR: ', TIOR, thresh_TIOR, config['Data_CLASSES'])


def train(visualizer, train_loader, model, optimizer, epoch, config):
    Cls_losses = AverageMeter()
    Ano_losses = AverageMeter()
    ExCls_losses = AverageMeter()
    Bound_losses = AverageMeter()
    Union_losses = AverageMeter()
    losses = AverageMeter()
    Cls_ACCs = AverageMeter()
    Ano_ACCs = AverageMeter()
    batch_time = AverageMeter()

    model.train()
    epoch_iter = 0
    num_classes = len(config['Data_CLASSES'])
    end = time.time()

    for i, (inputs, masks, cls_labels, flags, bbox_tags) in enumerate(train_loader):
        visualizer.reset()
        visual_ret = OrderedDict()
        errors_ret = OrderedDict()

        inputs = inputs.cuda()
        im_h = inputs.size(2)
        im_w = inputs.size(3)
        bs = inputs.size(0)
        visual_ret['input'] = inputs
        masks = masks.cuda()
        masks_vis = visual_masks(masks)
        visual_ret['mask'] = masks_vis
        cls_labels = cls_labels.cuda()
        ano_labels = torch.max(cls_labels, dim=1)[0]
        ano_labels = ano_labels.unsqueeze(1).float()
        flags = flags.cuda()
        bbox_tags = bbox_tags.cuda()

        outs_anomaly, outs_classes, cam_anomaly_refined, cam_classes_refined, cams_anomaly, cams_classes = model(inputs)

        outs_anomaly = torch.sigmoid(outs_anomaly)
        anoLoss = cls_criterion(outs_anomaly, ano_labels)

        outs_classes = torch.sigmoid(outs_classes)
        clsLoss = cls_criterion(outs_classes, cls_labels)

        # class-wise cam extra supervision
        exClsLosses = []
        boundLosses = []
        cams_cls_vis = []
        count1 = 0
        count2 = 0
        anomaly_cam_hidden = cams_anomaly.squeeze(1) * ano_labels.unsqueeze(2)
        anomaly_cam_mask = torch.sigmoid(cam_w*(anomaly_cam_hidden - cam_loss_sigma))
        cams_all = []
        for idx in range(num_classes):
            class_masks = masks[:, idx, :, :]
            class_cams = cam_classes_refined[:, idx, :, :]
            ex_cls_outs = ex_criterion(class_cams, class_masks)
            norm = ((class_cams.sum((1,2))+class_masks.sum((1,2)))*bbox_tags[:, idx]*flags.squeeze(1)).sum()
            ex_cls_outs = (ex_cls_outs.sum((1,2))*bbox_tags[:, idx]*flags.squeeze(1)).sum() / max(norm, 1e-5)
            exClsLosses.append(ex_cls_outs)
            class_cams_hidden = cams_classes[:, idx, :, :] * cls_labels[:, idx].unsqueeze(1).unsqueeze(2)
            cams_all.append(class_cams_hidden)
            class_cams_mask = torch.sigmoid(cam_w*(class_cams_hidden - cam_loss_sigma))
            bound_outs = ((class_cams_hidden.sum((1,2)) - (torch.min(class_cams_hidden, anomaly_cam_hidden)*class_cams_mask).sum((1,2))) / torch.clamp(class_cams_hidden.sum((1,2)), min=1e-5))*cls_labels[:, idx]
            norm = cls_labels[:, idx].sum()
            bound_outs = bound_outs.sum() / max(norm, 1e-5)
            boundLosses.append(bound_outs)
            class_cams = class_cams*bbox_tags[:, idx].unsqueeze(1).unsqueeze(2)
            class_cams = class_cams>=cam_thresh
            cams_cls_vis.append(class_cams.clone().unsqueeze(1))
            if (bbox_tags[:, idx]*flags.squeeze(1)).sum() > 0:
                count1 += 1
            if cls_labels[:, idx].sum() > 0:
                count2 += 1
        exClsLoss = sum(exClsLosses) / max(count1, 1)
        boundLoss = sum(boundLosses) / max(count2, 1)
        cams_all = torch.stack(cams_all, dim=1)
        cams_all = torch.max(cams_all, dim=1)[0]
        unionLoss = ((anomaly_cam_hidden.sum((1,2)) - (torch.min(cams_all, anomaly_cam_hidden)*anomaly_cam_mask).sum((1,2))) / torch.clamp((anomaly_cam_hidden).sum((1,2)), min=1e-5))*ano_labels.squeeze(1)
        norm = ano_labels.sum()
        unionLoss = unionLoss.sum() / max(norm, 1e-5)
        
        cams_cls_vis = torch.cat(cams_cls_vis, 1)
        cams_cls_vis = visual_masks(cams_cls_vis.float())
        visual_ret['attention'] = cams_cls_vis
        cam_anomaly_vis = cam_anomaly_refined*ano_labels.unsqueeze(2).unsqueeze(3)
        cam_anomaly_vis = cam_anomaly_vis>=cam_thresh
        cam_anomaly_vis = visual_masks(cam_anomaly_vis.float(), anomaly=True)
        visual_ret['anomaly'] = cam_anomaly_vis

        loss = loss_ano*anoLoss + loss_cls*clsLoss + loss_ex_cls*exClsLoss + loss_bound*boundLoss + loss_union*unionLoss

        errors_ret['anoLoss'] = loss_ano*float(anoLoss)
        errors_ret['clsLoss'] = loss_cls*float(clsLoss)
        errors_ret['exClsLoss'] = loss_ex_cls*float(exClsLoss)
        errors_ret['boundLoss'] = loss_bound*float(boundLoss)
        errors_ret['unionLoss'] = loss_union*float(unionLoss)
        errors_ret['Loss'] = float(loss)
        
        Ano_losses.update(loss_ano*anoLoss.item(), bs)
        Cls_losses.update(loss_cls*clsLoss.item(), bs)
        ExCls_losses.update(loss_ex_cls*exClsLoss.item(), bs)
        Bound_losses.update(loss_bound*boundLoss.item(), bs)
        Union_losses.update(loss_union*unionLoss.item(), bs)
        losses.update(loss.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m_acc_ano, _ = accuracy(outs_anomaly.detach(), ano_labels.detach(), 1)
        Ano_ACCs.update(m_acc_ano)
        m_acc_cls, _ = accuracy(outs_classes.detach(), cls_labels.detach(), num_classes)
        Cls_ACCs.update(m_acc_cls)
        epoch_iter += bs

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config['print_freq'] == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'anoLoss {anoloss.val:.4f} ({anoloss.avg:.4f})\t'
                  'clsLoss {clsloss.val:.4f} ({clsloss.avg:.4f})\t'
                  'exClsLoss {exclsloss.val:.4f} ({exclsloss.avg:.4f})\t'
                  'boundLoss {boundloss.val:.4f} ({boundloss.avg:.4f})\t'
                  'unionLoss {unionloss.val:.4f} ({unionloss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'anoACC {anoacc.val:.4f} ({anoacc.avg:.4f})\t'
                  'clsACC {clsacc.val:.4f} ({clsacc.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   anoloss=Ano_losses, clsloss=Cls_losses,
                   exclsloss=ExCls_losses, boundloss=Bound_losses, unionloss=Union_losses,
                   loss=losses, anoacc=Ano_ACCs, clsacc=Cls_ACCs))
            if config['display_id'] > 0:
                visualizer.plot_current_losses(epoch, 
                    float(epoch_iter) / float(config['train_length']), errors_ret)
        if i % config['display_freq'] == 0:
            visualizer.display_current_results(visual_ret, epoch, save_result=False)

def val(test_loader_cls, test_loader_iou, model, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    num_classes = len(config['Data_CLASSES'])

    with torch.no_grad():
        end = time.time()
        for i, (inputs, masks, cls_labels, flags, bbox_tags) in enumerate(test_loader_cls):
            inputs = inputs.cuda()
            bs = inputs.size(0)
            cls_labels = cls_labels.cuda()
            ano_labels = torch.max(cls_labels, dim=1)[0]
            ano_labels = ano_labels.unsqueeze(1).float()

            outs_anomaly, outs_classes, cam_anomaly_refined, cam_classes_refined, cams_anomaly, cams_classes = model(inputs)

            outs_anomaly = torch.sigmoid(outs_anomaly)
            anoLoss = cls_criterion(outs_anomaly, ano_labels)

            outs_classes = torch.sigmoid(outs_classes)
            clsLoss = cls_criterion(outs_classes, cls_labels)

            loss = loss_ano*anoLoss + loss_cls*clsLoss
            
            losses.update(loss.item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                logging.info('Test-Cls: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(test_loader_cls), batch_time=batch_time, loss=losses))

            if i == 0:
                y_gt_ano = ano_labels.detach().cpu().numpy()
                y_pred_ano = outs_anomaly.cpu().numpy()
                y_gt_cls = cls_labels.detach().cpu().numpy()
                y_pred_cls = outs_classes.cpu().numpy()
            else:
                y_gt_ano = np.concatenate((y_gt_ano, ano_labels.detach().cpu().numpy()), axis=0)
                y_pred_ano = np.concatenate((y_pred_ano, outs_anomaly.cpu().numpy()), axis=0)
                y_gt_cls = np.concatenate((y_gt_cls, cls_labels.detach().cpu().numpy()), axis=0)
                y_pred_cls = np.concatenate((y_pred_cls, outs_classes.cpu().numpy()), axis=0)

        auc_cls, all_auc_cls = calculate_auc(y_pred_cls, y_gt_cls, len(config['Data_CLASSES']))
        acc_cls, all_acc_cls = accuracy(torch.from_numpy(y_pred_cls).cuda(), 
                                        torch.from_numpy(y_gt_cls).cuda(), len(config['Data_CLASSES']))
        auc_ano, _ = calculate_auc(y_pred_ano, y_gt_ano, 1)
        acc_ano, _ = accuracy(torch.from_numpy(y_pred_ano).cuda(), 
                                        torch.from_numpy(y_gt_ano).cuda(), 1)
        del y_pred_cls
        del y_gt_cls
        del y_pred_ano
        del y_gt_ano
        gc.collect()

        end = time.time()
        epoch_iter = 0
        counts = np.zeros(num_classes)
        TIOU = np.zeros((num_classes, len(thresh_TIOU)))
        TIOR = np.zeros((num_classes, len(thresh_TIOR)))
        for i, (inputs, masks, cls_labels, flags, bbox_tags) in enumerate(test_loader_iou):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            masks = masks.cuda()
            masks_vis = visual_masks(masks)
            cls_labels = cls_labels.cuda()
            ano_labels = torch.max(cls_labels, dim=1)[0]
            ano_labels = ano_labels.unsqueeze(1).float()
            flags = flags.cuda()
            bbox_tags = bbox_tags.cuda()

            outs_anomaly, outs_classes, cam_anomaly_refined, cam_classes_refined, cams_anomaly, cams_classes = model(inputs)

            outs_anomaly = torch.sigmoid(outs_anomaly)
            anoLoss = cls_criterion(outs_anomaly, ano_labels)

            outs_classes = torch.sigmoid(outs_classes)
            clsLoss = cls_criterion(outs_classes, cls_labels)

            # class-wise cam extra supervision
            exClsLosses = []
            boundLosses = []
            cams_cls_vis = []
            count1 = 0
            count2 = 0
            anomaly_cam_hidden = cams_anomaly.squeeze(1) * ano_labels.unsqueeze(2)
            anomaly_cam_mask = torch.sigmoid(cam_w*(anomaly_cam_hidden - cam_loss_sigma))
            cams_all = []
            for idx in range(num_classes):
                class_masks = masks[:, idx, :, :]
                class_cams = cam_classes_refined[:, idx, :, :]
                ex_cls_outs = ex_criterion(class_cams, class_masks)
                norm = ((class_cams.sum((1,2))+class_masks.sum((1,2)))*bbox_tags[:, idx]*flags.squeeze(1)).sum()
                ex_cls_outs = (ex_cls_outs.sum((1,2))*bbox_tags[:, idx]*flags.squeeze(1)).sum() / max(norm, 1e-5)
                exClsLosses.append(ex_cls_outs)
                class_cams_hidden = cams_classes[:, idx, :, :] * cls_labels[:, idx].unsqueeze(1).unsqueeze(2)
                cams_all.append(class_cams_hidden)
                class_cams_mask = torch.sigmoid(cam_w*(class_cams_hidden - cam_loss_sigma))
                bound_outs = ((class_cams_hidden.sum((1,2)) - (torch.min(class_cams_hidden, anomaly_cam_hidden)*class_cams_mask).sum((1,2))) / torch.clamp(class_cams_hidden.sum((1,2)), min=1e-5))*cls_labels[:, idx]
                norm = cls_labels[:, idx].sum()
                bound_outs = bound_outs.sum() / max(norm, 1e-5)
                boundLosses.append(bound_outs)
                if (bbox_tags[:, idx]*flags.squeeze(1)).sum() > 0:
                    count1 += 1
                if cls_labels[:, idx].sum() > 0:
                    count2 += 1
            exClsLoss = sum(exClsLosses) / max(count1, 1)
            boundLoss = sum(boundLosses) / max(count2, 1)
            cams_all = torch.stack(cams_all, dim=1)
            cams_all = torch.max(cams_all, dim=1)[0]
            cams_all_mask = torch.sigmoid(cam_w*(cams_all - cam_loss_sigma))
            unionLoss = (((anomaly_cam_hidden+cams_all).sum((1,2)) - 2*(torch.min(cams_all, anomaly_cam_hidden)*cams_all_mask*anomaly_cam_mask).sum((1,2))) / torch.clamp((anomaly_cam_hidden+cams_all).sum((1,2)), min=1e-5))*ano_labels.squeeze(1)
            norm = ano_labels.sum()
            unionLoss = unionLoss.sum() / max(norm, 1e-5)

            loss = loss_ano*anoLoss + loss_cls*clsLoss + loss_ex_cls*exClsLoss + loss_bound*boundLoss + loss_union*unionLoss
            
            losses.update(loss.item(), bs)

            cam_classes_refined = cam_classes_refined>=cam_thresh

            epoch_iter += bs

            for idx in range(num_classes):
                if bbox_tags[0][idx] == 1:
                    batch_iou = single_IOU(cam_classes_refined[:, idx, :, :], masks[:, idx, :, :])
                    batch_ior = single_IOR(cam_classes_refined[:, idx, :, :], masks[:, idx, :, :])
                    for j in range(len(thresh_TIOU)):
                        if batch_iou >= thresh_TIOU[j]:
                            TIOU[idx][j] += 1
                    for j in range(len(thresh_TIOR)):
                        if batch_ior >= thresh_TIOR[j]:
                            TIOR[idx][j] += 1
                    counts[idx] += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                logging.info('Test-IoU: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(test_loader_iou), batch_time=batch_time, loss=losses))

        for idx in range(num_classes):
            for j in range(len(thresh_TIOU)):
                if counts[idx]==0:
                    TIOU[idx][j] = 0.
                else:
                    TIOU[idx][j] = float(TIOU[idx][j])/float(counts[idx])
        for idx in range(num_classes):
            for j in range(len(thresh_TIOR)):
                if counts[idx]==0:
                    TIOR[idx][j] = 0.
                else:
                    TIOR[idx][j] = float(TIOR[idx][j])/float(counts[idx])

        return TIOU, TIOR, auc_cls, all_auc_cls, acc_cls, all_acc_cls, auc_ano, acc_ano


def test(test_loader_cls, test_loader_iou, model, config, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    num_classes = len(config['Data_CLASSES'])
    preds_att = torch.zeros(config['val_length_iou'], num_classes, config['img_size'], config['img_size'])
    masks_all = torch.zeros(config['val_length_iou'], num_classes, config['img_size'], config['img_size'])
    preds_path = os.path.join(outputs_path, 'test', args.config.split('/')[-1].split('.')[0]+'-yaml')
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)
    img_list = []
    for line in open(os.path.join(config['DataRoot'], 'ImageSets', 'bbox', config['testIoUSet'])):
        img_list.append(line.strip())

    with torch.no_grad():
        end = time.time()
        for i, (inputs, masks, cls_labels, flags, bbox_tags) in enumerate(test_loader_cls):
            inputs = inputs.cuda()
            bs = inputs.size(0)
            cls_labels = cls_labels.cuda()
            ano_labels = torch.max(cls_labels, dim=1)[0]
            ano_labels = ano_labels.unsqueeze(1).float()

            outs_anomaly, outs_classes, cam_anomaly_refined, cam_classes_refined, cams_anomaly, cams_classes = model(inputs)

            outs_anomaly = torch.sigmoid(outs_anomaly)
            anoLoss = cls_criterion(outs_anomaly, ano_labels)

            outs_classes = torch.sigmoid(outs_classes)
            clsLoss = cls_criterion(outs_classes, cls_labels)

            loss = loss_ano*anoLoss + loss_cls*clsLoss
            
            losses.update(loss.item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                logging.info('Test-Cls: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(test_loader_cls), batch_time=batch_time, loss=losses))

            if i == 0:
                y_gt_ano = ano_labels.detach().cpu().numpy()
                y_pred_ano = outs_anomaly.cpu().numpy()
                y_gt_cls = cls_labels.detach().cpu().numpy()
                y_pred_cls = outs_classes.cpu().numpy()
            else:
                y_gt_ano = np.concatenate((y_gt_ano, ano_labels.detach().cpu().numpy()), axis=0)
                y_pred_ano = np.concatenate((y_pred_ano, outs_anomaly.cpu().numpy()), axis=0)
                y_gt_cls = np.concatenate((y_gt_cls, cls_labels.detach().cpu().numpy()), axis=0)
                y_pred_cls = np.concatenate((y_pred_cls, outs_classes.cpu().numpy()), axis=0)

        y_gt_ano_file = open(os.path.join(preds_path, 'y_gt_ano.pkl'), 'wb')
        y_pred_ano_file = open(os.path.join(preds_path, 'y_pred_ano.pkl'), 'wb')
        y_gt_cls_file = open(os.path.join(preds_path, 'y_gt_cls.pkl'), 'wb')
        y_pred_cls_file = open(os.path.join(preds_path, 'y_pred_cls.pkl'), 'wb')
        pickle.dump(y_gt_ano, y_gt_ano_file)
        pickle.dump(y_pred_ano, y_pred_ano_file)
        pickle.dump(y_gt_cls, y_gt_cls_file)
        pickle.dump(y_pred_cls, y_pred_cls_file)

        auc_cls, all_auc_cls = calculate_auc(y_pred_cls, y_gt_cls, len(config['Data_CLASSES']))
        acc_cls, all_acc_cls = accuracy(torch.from_numpy(y_pred_cls).cuda(), 
                                        torch.from_numpy(y_gt_cls).cuda(), len(config['Data_CLASSES']))
        auc_ano, _ = calculate_auc(y_pred_ano, y_gt_ano, 1)
        acc_ano, _ = accuracy(torch.from_numpy(y_pred_ano).cuda(), 
                                        torch.from_numpy(y_gt_ano).cuda(), 1)
        del y_pred_cls
        del y_gt_cls
        del y_pred_ano
        del y_gt_ano
        gc.collect()

        end = time.time()
        epoch_iter = 0
        counts = np.zeros(num_classes)
        TIOU = np.zeros((num_classes, len(thresh_TIOU)))
        TIOR = np.zeros((num_classes, len(thresh_TIOR)))
        bbox_tags_dict = {}
        for i, (inputs, masks, cls_labels, flags, bbox_tags) in enumerate(test_loader_iou):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            masks = masks.cuda()
            cls_labels = cls_labels.cuda()
            ano_labels = torch.max(cls_labels, dim=1)[0]
            ano_labels = ano_labels.unsqueeze(1).float()
            flags = flags.cuda()
            bbox_tags = bbox_tags.cuda()
            bbox_tags_save = bbox_tags.cpu().numpy()
            img_name = img_list[i].split('.')[0]
            bbox_tags_dict[img_name] = bbox_tags_save

            outs_anomaly, outs_classes, cam_anomaly_refined, cam_classes_refined, cams_anomaly, cams_classes = model(inputs)

            outs_anomaly = torch.sigmoid(outs_anomaly)
            anoLoss = cls_criterion(outs_anomaly, ano_labels)

            outs_classes = torch.sigmoid(outs_classes)
            clsLoss = cls_criterion(outs_classes, cls_labels)

            # class-wise cam extra supervision
            exClsLosses = []
            boundLosses = []
            cams_cls_vis = []
            count1 = 0
            count2 = 0
            anomaly_cam_hidden = cams_anomaly.squeeze(1) * ano_labels.unsqueeze(2)
            anomaly_cam_mask = torch.sigmoid(cam_w*(anomaly_cam_hidden - cam_loss_sigma))
            cams_all = []
            for idx in range(num_classes):
                class_masks = masks[:, idx, :, :]
                class_cams = cam_classes_refined[:, idx, :, :]
                ex_cls_outs = ex_criterion(class_cams, class_masks)
                norm = ((class_cams.sum((1,2))+class_masks.sum((1,2)))*bbox_tags[:, idx]*flags.squeeze(1)).sum()
                ex_cls_outs = (ex_cls_outs.sum((1,2))*bbox_tags[:, idx]*flags.squeeze(1)).sum() / max(norm, 1e-5)
                exClsLosses.append(ex_cls_outs)
                class_cams_hidden = cams_classes[:, idx, :, :] * cls_labels[:, idx].unsqueeze(1).unsqueeze(2)
                cams_all.append(class_cams_hidden)
                class_cams_mask = torch.sigmoid(cam_w*(class_cams_hidden - cam_loss_sigma))
                bound_outs = ((class_cams_hidden.sum((1,2)) - (torch.min(class_cams_hidden, anomaly_cam_hidden)*class_cams_mask).sum((1,2))) / torch.clamp(class_cams_hidden.sum((1,2)), min=1e-5))*cls_labels[:, idx]
                norm = cls_labels[:, idx].sum()
                bound_outs = bound_outs.sum() / max(norm, 1e-5)
                boundLosses.append(bound_outs)
                if (bbox_tags[:, idx]*flags.squeeze(1)).sum() > 0:
                    count1 += 1
                if cls_labels[:, idx].sum() > 0:
                    count2 += 1
            exClsLoss = sum(exClsLosses) / max(count1, 1)
            boundLoss = sum(boundLosses) / max(count2, 1)
            cams_all = torch.stack(cams_all, dim=1)
            cams_all = torch.max(cams_all, dim=1)[0]
            cams_all_mask = torch.sigmoid(cam_w*(cams_all - cam_loss_sigma))
            unionLoss = (((anomaly_cam_hidden+cams_all).sum((1,2)) - 2*(torch.min(cams_all, anomaly_cam_hidden)*cams_all_mask*anomaly_cam_mask).sum((1,2))) / torch.clamp((anomaly_cam_hidden+cams_all).sum((1,2)), min=1e-5))*ano_labels.squeeze(1)
            norm = ano_labels.sum()
            unionLoss = unionLoss.sum() / max(norm, 1e-5)

            loss = loss_ano*anoLoss + loss_cls*clsLoss + loss_ex_cls*exClsLoss + loss_bound*boundLoss + loss_union*unionLoss
            
            losses.update(loss.item(), bs)

            cam_classes_refined = cam_classes_refined>=cam_thresh

            epoch_iter += bs

            for idx in range(num_classes):
                if bbox_tags[0][idx] == 1:
                    batch_iou = single_IOU(cam_classes_refined[:, idx, :, :], masks[:, idx, :, :])
                    batch_ior = single_IOR(cam_classes_refined[:, idx, :, :], masks[:, idx, :, :])
                    for j in range(len(thresh_TIOU)):
                        if batch_iou >= thresh_TIOU[j]:
                            TIOU[idx][j] += 1
                    for j in range(len(thresh_TIOR)):
                        if batch_ior >= thresh_TIOR[j]:
                            TIOR[idx][j] += 1
                    counts[idx] += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                logging.info('Test-IoU: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(test_loader_iou), batch_time=batch_time, loss=losses))

        TIOU_file = open(os.path.join(preds_path, 'TIOU.pkl'), 'wb')
        TIOR_file = open(os.path.join(preds_path, 'TIOR.pkl'), 'wb')
        Counts_file = open(os.path.join(preds_path, 'Counts.pkl'), 'wb')
        bbox_tags_file = open(os.path.join(preds_path, 'bbox_tags.pkl'), 'wb')
        pickle.dump(TIOU, TIOU_file)
        pickle.dump(TIOR, TIOR_file)
        pickle.dump(counts, Counts_file)
        pickle.dump(bbox_tags_dict, bbox_tags_file)

        for idx in range(num_classes):
            for j in range(len(thresh_TIOU)):
                if counts[idx]==0:
                    TIOU[idx][j] = 0.
                else:
                    TIOU[idx][j] = float(TIOU[idx][j])/float(counts[idx])
        for idx in range(num_classes):
            for j in range(len(thresh_TIOR)):
                if counts[idx]==0:
                    TIOR[idx][j] = 0.
                else:
                    TIOR[idx][j] = float(TIOR[idx][j])/float(counts[idx])

        return TIOU, TIOR, auc_cls, all_auc_cls, acc_cls, all_acc_cls, auc_ano, acc_ano


def visual(visual_loader, model, config, args):
    save_path = os.path.join(outputs_path, 'visual', args.config.split('/')[-1].split('.')[0]+'-yaml')
    imgs_path = os.path.join(save_path, 'images')
    preds_path = os.path.join(save_path, 'preds')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)
    img_list = []
    for line in open(os.path.join(config['DataRoot'], 'ImageSets', 'bbox', config['testIoUSet'])):
        img_list.append(line.strip())
    bbox_gts = json.load(open(os.path.join(config['DataRoot'], 'Annotations', 'BBoxes.json')))
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    num_classes = len(config['Data_CLASSES'])
    webpage = html.HTML(save_path, 'Experiment name = visualization', reflesh=1)

    with torch.no_grad():
        end = time.time()
        counts = np.zeros(num_classes)
        TIOU = np.zeros((num_classes, len(thresh_TIOU)))
        TIOR = np.zeros((num_classes, len(thresh_TIOR)))
        bbox_tags_dict = {}
        for i, (inputs, masks, cls_labels, flags, bbox_tags) in enumerate(visual_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            masks = masks.cuda()
            cls_labels = cls_labels.cuda()
            ano_labels = torch.max(cls_labels, dim=1)[0]
            ano_labels = ano_labels.unsqueeze(1).float()
            flags = flags.cuda()
            bbox_tags = bbox_tags.cuda()
            bbox_tags_save = bbox_tags.cpu().numpy()

            img_name = img_list[i].split('.')[0]
            bbox_img_gt = bbox_gts[img_name+'.png']
            bbox_tags_dict[img_name] = bbox_tags_save
            webpage.add_header('{}'.format(img_list[i]))
            ims, txts, links = [], [], []
            diseases = ''
            for idx in range(num_classes):
                if bbox_tags[0][idx] == 1:
                    diseases += config['Data_CLASSES'][idx]
                    diseases += ', '

            outs_anomaly, outs_classes, cam_anomaly_refined, cam_classes_refined, cams_anomaly, cams_classes = model(inputs)


            cam_classes_vis = []
            for idx in range(num_classes):
                class_gcams = cam_classes_refined[:, idx, :, :]
                class_gcams = class_gcams*bbox_tags[:, idx].unsqueeze(1).unsqueeze(1)
                cam_classes_vis.append(class_gcams.clone().unsqueeze(1))
                if idx == 0:
                    erasing_mask = class_gcams>=cam_thresh
                else:
                    erasing_mask += class_gcams>=cam_thresh
            cam_classes_vis = torch.cat(cam_classes_vis, 1)
            cam_classes_vis = cam_classes_vis>=cam_thresh
            cam_classes_vis = visual_masks(cam_classes_vis.float())

            cam_anomaly_vis = cam_anomaly_refined*ano_labels.unsqueeze(1).unsqueeze(1)
            cam_anomaly_vis = cam_anomaly_vis>=cam_thresh
            cam_anomaly_vis = visual_masks(cam_anomaly_vis.float(), anomaly=True)

            # save images
            image_numpy = vis_util.tensor2im('input', inputs)
            img_path_img = os.path.join(imgs_path, img_name+'.png')
            ims.append(img_name+'.png')
            txts.append('Input')
            links.append(img_name+'.png')
            vis_util.save_image(image_numpy, img_path_img)

            bbox_img = cv2.imread(img_path_img)
            bbox_path = os.path.join(imgs_path, img_name+'_bbox.png')
            ims.append(img_name+'_bbox.png')
            txts.append('BBox: '+diseases[:-2])
            links.append(img_name+'_bbox.png')
            factor = float(config['img_size']) / 1024.
            for b_i in range(len(config['Data_CLASSES'])):
                if config['Data_CLASSES'][b_i] in bbox_img_gt:
                    for loc in bbox_img_gt[config['Data_CLASSES'][b_i]]:
                        bbox_img = cv2.rectangle(bbox_img, (int(loc[0]*factor), int(loc[1]*factor)), (int((loc[0]+loc[2])*factor), int((loc[1]+loc[3])*factor)), (palette[b_i+1][2],palette[b_i+1][1],palette[b_i+1][0]), 4)
            cv2.imwrite(bbox_path, bbox_img)

            masks_vis = visual_masks(masks)
            image_numpy = vis_util.tensor2im('mask', masks_vis)
            img_path_mask = os.path.join(imgs_path, img_name+'_mask.png')
            ims.append(img_name+'_mask.png')
            txts.append('Mask: '+diseases[:-2])
            links.append(img_name+'_mask.png')
            vis_util.save_image(image_numpy, img_path_mask)

            img = cv2.imread(img_path_img)
            gts = cv2.imread(img_path_mask)
            combined = combine_img(img, gts)
            img_path = os.path.join(imgs_path, img_name+'_mask_img.png')
            ims.append(img_name+'_mask_img.png')
            txts.append('Mask-Img')
            links.append(img_name+'_mask_img.png')
            cv2.imwrite(img_path, combined)

            image_numpy = vis_util.tensor2im('attention', cam_anomaly_vis)
            img_path_cams = os.path.join(imgs_path, img_name+'_anomaly_attention.png')
            ims.append(img_name+'_anomaly_attention.png')
            txts.append('Anomaly Attention')
            links.append(img_name+'_anomaly_attention.png')
            vis_util.save_image(image_numpy, img_path_cams)

            img = cv2.imread(bbox_path)
            cams = cv2.imread(img_path_cams)
            combined = combine_img(img, cams)
            img_path = os.path.join(imgs_path, img_name+'_anomaly_img.png')
            ims.append(img_name+'_anomaly_img.png')
            txts.append('Anomaly-Img')
            links.append(img_name+'_anomaly_img.png')
            cv2.imwrite(img_path, combined)

            image_numpy = vis_util.tensor2im('attention', cam_classes_vis)
            img_path_cams = os.path.join(imgs_path, img_name+'_disease_attention.png')
            ims.append(img_name+'_disease_attention.png')
            txts.append('Disease Attention')
            links.append(img_name+'_disease_attention.png')
            vis_util.save_image(image_numpy, img_path_cams)
            
            img = cv2.imread(bbox_path)
            cams = cv2.imread(img_path_cams)
            combined = combine_img(img, cams)
            img_path = os.path.join(imgs_path, img_name+'_disease_img.png')
            ims.append(img_name+'_disease_img.png')
            txts.append('Disease-Img')
            links.append(img_name+'_disease_img.png')
            cv2.imwrite(img_path, combined)

            webpage.add_images(ims, txts, links, width=config['display_winsize'])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                logging.info('Visual: [{0}/{1}]\t'
                      'Image {2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       i, len(visual_loader), img_list[i], batch_time=batch_time))

        bbox_tags_file = open(os.path.join(preds_path, 'bbox_tags.pkl'), 'wb')
        pickle.dump(bbox_tags_dict, bbox_tags_file)
        webpage.save()


def single_IOU(pred, target):
    pred_class = pred.data.cpu().contiguous().view(-1)
    target_class = target.data.cpu().contiguous().view(-1)
    pred_inds = pred_class == 1
    target_inds = target_class == 1
    intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    iou = float(intersection) / float(max(union, 1))
    return iou

def single_IOR(pred, target):
    pred_class = pred.data.cpu().contiguous().view(-1)
    target_class = target.data.cpu().contiguous().view(-1)
    pred_inds = pred_class == 1
    target_inds = target_class == 1
    intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    iou = float(intersection) / float(max(pred_inds.long().sum().item(), 1))
    return iou

def visual_masks(masks, anomaly=False):
    mask_vis = masks.clone()
    bs = mask_vis.size(0)
    num_classes = mask_vis.size(1)
    im_h = mask_vis.size(2)
    im_w = mask_vis.size(3)
    mask_one = torch.zeros((bs, im_h, im_w)).cuda()
    for idx in range(num_classes):
        mask_one = mask_one + mask_vis[:, idx, :, :]*(idx+1)
        mask_one[mask_one>idx+1] = idx+1
    vis_mask1 = mask_one.clone()
    vis_mask2 = mask_one.clone()
    vis_mask3 = mask_one.clone()
    if anomaly:
        vis_mask1[vis_mask1==1] = palette[-1][0]
        vis_mask2[vis_mask2==1] = palette[-1][1]
        vis_mask3[vis_mask3==1] = palette[-1][2]
    else:
        for idx in range(num_classes+1):
            vis_mask1[vis_mask1==idx] = palette[idx][0]
            vis_mask2[vis_mask2==idx] = palette[idx][1]
            vis_mask3[vis_mask3==idx] = palette[idx][2]
    vis_mask1 = vis_mask1.unsqueeze(1)
    vis_mask2 = vis_mask2.unsqueeze(1)
    vis_mask3 = vis_mask3.unsqueeze(1)
    vis_mask = torch.cat((vis_mask1, vis_mask2, vis_mask3), 1)
    return vis_mask

def combine_img(img, cams, flag=0):
    binary_mask = torch.from_numpy(cams)
    binary_mask = binary_mask.sum(dim=2).unsqueeze(2)
    binary_mask = binary_mask >= 1
    binary_mask = binary_mask.float()
    img = torch.from_numpy(img).float()
    img_out = ((1 - binary_mask) * img).numpy()
    img_out = img_out.astype(np.uint8)
    img_in = (binary_mask * img).numpy()
    img_in = img_in.astype(np.uint8)
    img_in = cv2.addWeighted(img_in, 0.65, cams, 0.35, 0)
    outs = img_out + img_in
    return outs
