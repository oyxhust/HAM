import os
import logging
import torch.nn as nn
from .AttentionModel.resnet import *
import torchvision.models as models
from modules.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class AttentionNet(nn.Module):
    """
    Build Attention Model
    """
    def __init__(self, model_arch, stride_list, dilations, using_pooling, num_classes):
        super(AttentionNet, self).__init__()
        self.arch = model_arch
        self.model = eval(model_arch)(stride_list, using_pooling, dilations, norm_layer=SynchronizedBatchNorm2d, num_classes=num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def load_pretrained_weights(self):
        logging.info('Loading pretrained weights...')
        pretrained_dict = models.__dict__[self.arch](pretrained=True).state_dict()
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        logging.info('Finished loading pretrained weights!')


def build_model(arch='resnet18', downsampling=32, using_pooling=True, using_dilation=False, num_classes=14):
    logging.info("Now supporting resnet for Attention")

    if using_pooling==True:
        if downsampling==32:
            stride_list = [1, 2, 2, 2]
            if using_dilation==False:
                dilations = None
            else:
                dilations = (1, 1, 1, 2)
        elif downsampling==16:
            stride_list = [1, 2, 2, 1]
            if using_dilation==False:
                dilations = None
            else:
                dilations = (1, 1, 1, 2)
        elif downsampling==8:
            stride_list = [1, 2, 1, 1]
            if using_dilation==False:
                dilations = None
            else:
                dilations = (1, 1, 2, 4)
    else:
        if downsampling==32:
            stride_list = [2, 2, 2, 2]
            if using_dilation==False:
                dilations = None
            else:
                dilations = (1, 1, 1, 2)
        elif downsampling==16:
            stride_list = [1, 2, 2, 2]
            if using_dilation==False:
                dilations = None
            else:
                dilations = (1, 1, 2, 4)
        elif downsampling==8:
            stride_list = [1, 2, 2, 1]
            if using_dilation==False:
                dilations = None
            else:
                dilations = (1, 2, 4, 6)

    return AttentionNet(arch, stride_list, dilations, using_pooling, num_classes)