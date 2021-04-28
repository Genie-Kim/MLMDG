import argparse
from segmodel import *
from pprint import pprint
from TSMLDG import MetaFrameWork

from pathlib import Path
from dataset.dg_dataset import get_target_loader
import numpy as np

import os

parser = argparse.ArgumentParser(description='TSMLDG train args parser')
parser.add_argument('--name', default='exp', help='name of the experiment')
parser.add_argument('--source', default='GSIM', help='source domain name list, capital of the first character of dataset "GSIMcuv"(dataset should exists first.)')
parser.add_argument('--target', default='C', help='target domain name, only one target supported')

parser.add_argument('--resume', action='store_true', help='resume the training procedure')
parser.add_argument('--debug', action='store_true', help='set the workers=0 and batch size=1 to accelerate debug')

parser.add_argument('--inner-lr', type=float, default=1e-3, help='inner learning rate of meta update')
parser.add_argument('--outer-lr', type=float, default=5e-3, help='outer learning rate of network update')
parser.add_argument('--no_outer_memloss', action='store_true', help='memory loss on outer update step')
parser.add_argument('--no_inner_memloss', action='store_true', help='memory loss on outer update step')
parser.add_argument('--lamb_cpt', type=float, default=1e-1, help='inner learning rate of meta update')
parser.add_argument('--lamb_sep', type=float, default=1e-2, help='outer learning rate of network update')
parser.add_argument('--train-size', type=int, default=2, help='the batch size of training')
parser.add_argument('--test-size', type=int, default=2, help='the batch size of evaluation')
parser.add_argument('--train-num', type=int, default=1,
                    help='every ? iteration do one meta train, 1 is meta train, 10000000 is normal supervised learning.')

parser.add_argument('--network', default='Deeplabv3plus_Memsup', help='network for DG')
parser.add_argument('--memory', action='store_true', help='use memory')
parser.add_argument('--supervised_mem', action='store_true', help='use supervised memory')
parser.add_argument('--add1by1', action='store_true', help='use supervised memory')
parser.add_argument('--sche', default='lrplate',choices=['lrplate', 'cosine', 'poly'], help='scheduler')

# parser.add_argument('--configs', default='MemDeeplabv3plus', help='configfile')



def init():
    args = vars(parser.parse_args())
    if args['debug'] == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        args['name'] = 'tmp'
    # if 'configs' in args:
    #     config_file = args['configs']
    #     import json
    #     args.update(json.loads(config_file))


    for name in args['source']:
        assert name in 'GSIMCcuv'
    for name in args['target']:
        assert name in 'GSIMCcuv'


    if args['network'] in ['Deeplabv3plus_Memsup','Deeplabv3plus_Memunsup']:
        if args.pop('memory'):
            supervised_mem = args.pop('supervised_mem')
            memory_init = {'memory_size': 19 if supervised_mem else 19, 'feature_dim': 256, 'key_dim': 256, 'temp_update': 0.1,
                           'temp_gather': 0.1,'supervised_mem':supervised_mem,'momentum' : 0.2,'temperature' : 0.1}
        else:
            args.pop('supervised_mem')
            memory_init = None

        network_init = {'in_ch': 3, 'nclass': 19, 'backbone': 'resnet50', 'output_stride': 8, 'pretrained': True,
                        'freeze_bn': False, 'add1by1': args.pop('add1by1'),
                        'freeze_backbone': False, 'skipconnect': True, 'memory_init': memory_init}

    args.update({'network_init': network_init})

    pprint(args)

    return args

def train():
    args = init()
    assert len(args['target'])
    framework = MetaFrameWork(**args)
    framework.do_train()

def draw_tsne():
    args = init()
    framework = MetaFrameWork(**args)
    framework.draw_tsne_domcls(perplexities=[30], learning_rate=10, imagenum=1,pthname = 'best_city')


def predict():
    args = init()
    framework = MetaFrameWork(**args)
    framework.predict_target(load_path='best_city', output_path='predictions',savenum = 3,inputimgname=None)

def eval():
    args = init()
    names = [args['name']]
    targets = args['target']
    args['target'] = 'C'
    args['name'] = 'tmp'
    framework = MetaFrameWork(**args)
    for name, target in zip(names, targets):
        # for one experiment, test multi targets in multi batch_sizes
        framework.print('=' * 20 + ' {} '.format(name) + '=' * 20)
        framework.save_path = Path('experiment/'+name)

        # framework.load('best_city', strict=False)
        framework.load('ckpt', strict=False)

        seeds = [123456]
        for target in targets:
            for i, seed in enumerate(seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                framework.log('-' * 20 + ' {} '.format(target) + '-' * 20 + 'seed : '+str(seed) + '\n\n')
                framework.val(get_target_loader(target, args['test_size'], mode='test',shuffle=False))

if __name__ == '__main__':
    # from utils.task import FunctionJob
    # job = FunctionJob([train], gpus=[[1]])
    # job.run(minimum_memory=10000)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
    # draw_tsne()
    # predict()
    # eval()