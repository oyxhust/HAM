import os
import argparse
import yaml
import logging
import torch
import torch.backends.cudnn as cudnn
from utils.logging import open_log
from utils.tools import load_checkpoint
from utils.visualizer import Visualizer
from models import AttentionNet
from utils import net_utils


def arg_parse():
    parser = argparse.ArgumentParser(
        description='AttentionNet')
    parser.add_argument('-cfg', '--config', default='configs/config.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('--use_html', default=True,
                        type=bool, help='Use html')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # open log file
    open_log(args)
    logging.info(args)
    logging.info(config)
    visualizer = Visualizer('AttentionNet', config, args)

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # set net
    AttentionModel = AttentionNet.build_model(config['arch'], config['Downsampling'], config['Using_pooling'], config['Using_dilation'], len(config['Data_CLASSES']))
    if config['Using_pretrained_weights']:
        AttentionModel.load_pretrained_weights()
    if config['Attention']['resume'] != None:
        load_checkpoint(AttentionModel, config['Attention']['resume'])

    if args.cuda:
        AttentionModel.cuda()
        cudnn.benchmark = True

    AttentionModel = torch.nn.DataParallel(AttentionModel)

    optimizer, train_loader, val_loader = net_utils.prepare_net(config, AttentionModel)

    net_utils.train_net(visualizer, optimizer, train_loader, val_loader, AttentionModel, config)

if __name__ == '__main__':
    main()