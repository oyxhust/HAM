import os
import argparse
import yaml
import logging
import torch
import torch.backends.cudnn as cudnn
from utils.logging import open_log
from utils.tools import load_checkpoint
from models import AttentionNet
from utils import net_utils


def arg_parse():
    parser = argparse.ArgumentParser(
        description='AttentionNet')
    parser.add_argument('-cfg', '--config', default='configs/config.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('-w', '--weight', default='',
                        type=str, help='load the model weight')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # open log file
    open_log(args, 'test')
    logging.info(args)
    logging.info(config)

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # set net
    AttentionModel = AttentionNet.build_model(config['arch'], config['Downsampling'], config['Using_pooling'], config['Using_dilation'], len(config['Data_CLASSES']))  
    assert args.weight != '' # must load a trained model
    logging.info('Resuming network: {}'.format(args.weight))
    load_checkpoint(AttentionModel, args.weight)

    if args.cuda:
        AttentionModel.cuda()

    AttentionModel = torch.nn.DataParallel(AttentionModel)

    test_loader = net_utils.prepare_net(config, AttentionModel, 'test')

    net_utils.test_net(test_loader, AttentionModel, config, args)


if __name__ == '__main__':
    main()
