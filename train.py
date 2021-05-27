import argparse
from segmodel import *
from pprint import pprint
from TSMLDG import MetaFrameWork

from pathlib import Path
from dataset.dg_dataset import get_target_loader
from dataset.transforms import FixScale
import numpy as np

import os

parser = argparse.ArgumentParser(description='TSMLDG train args parser')
parser.add_argument('--name', default='exp', help='name of the experiment')
parser.add_argument('--source', default='GS', help='source domain name list, capital of the first character of dataset "GSIMcuv"(dataset should exists first.)')
parser.add_argument('--target', default='C', help='target domain name, only one target supported')

parser.add_argument('--resume', action='store_true', help='resume the training procedure')
parser.add_argument('--debug', action='store_true', help='set the workers=0 and batch size=1 to accelerate debug')

parser.add_argument('--inner-lr', type=float, default=1e-3, help='inner learning rate of meta update')
parser.add_argument('--outer-lr', type=float, default=5e-3, help='outer learning rate of network update')
parser.add_argument('--no_outer_memloss', action='store_true', help='memory loss on outer update step')
parser.add_argument('--no_inner_memloss', action='store_true', help='memory loss on outer update step')
parser.add_argument('--lamb_cpt', type=float, default=2e-1, help='loss rate of reading contrastive loss')
parser.add_argument('--lamb_sep', type=float, default=1e-1, help='loss rate of writing losses')
parser.add_argument('--train-size', type=int, default=4, help='the batch size of training')
parser.add_argument('--test-size', type=int, default=1, help='the batch size of evaluation')
parser.add_argument('--train-num', type=int, default=1,
                    help='every ? iteration do one meta train, 1 is meta train, 10000000 is normal supervised learning.')

parser.add_argument('--network', default='Deeplabv3plus_Memsup', help='network for DG')
parser.add_argument('--backbone', default='resnet50',choices=['resnet50', 'resnet101'], help='network backbone')
# parser.add_argument('--output_stride', type=int, default=16, help='output stride of resnet backbone')
parser.add_argument('--memory', action='store_true', help='using memory module')
parser.add_argument('--add1by1', action='store_true', help='adding 1x1 conv for writing memory feature')
parser.add_argument('--clsfy_loss', action='store_true', help='using classification loss for memory')
parser.add_argument('--gumbel_read', action='store_true', help='using gumbel softmax function for reading')
parser.add_argument('--hideandseek', action='store_true', help='using hide and seek on writing')
parser.add_argument('--sche', default='lrplate',choices=['lrplate', 'cosine', 'poly'], help='scheduler')
parser.add_argument('--momentum', type=float, default=0.8, help='outer learning rate of network update')
parser.add_argument('--temperature', type=float, default=1, help='temperature of reading contrastive loss')

def init():
    args = vars(parser.parse_args())
    if args['debug'] == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        args['name'] = 'tmp'

    for name in args['source']:
        assert name in 'GSIMCcuv'
    for name in args['target']:
        assert name in 'GSIMCcuv'


    if args['network'] in ['Deeplabv3plus_Memsup']:

        if args['memory']:
            memory_init = {'memory_size': 19, 'feature_dim': 256,
                           'momentum': args['momentum'], 'temperature': args['temperature'], 'add1by1': args['add1by1'],
                           'clsfy_loss': args['clsfy_loss'], 'gumbel_read': args['gumbel_read']}
        else:
            memory_init = None

        network_init = {'in_ch': 3, 'nclass': 19, 'backbone': args['backbone'], 'output_stride': 16,
                        'pretrained': True,
                        'freeze_bn': False,
                        'freeze_backbone': False, 'skip_connect': True, 'hideandseek': args['hideandseek'],
                        'memory_init': memory_init}


    elif args['network'] in ['FCN_Memsup']:

        if args['memory']:
            memory_init = {'memory_size': 19, 'feature_dim': 256,
                           'momentum': args['momentum'], 'temperature': args['temperature'], 'add1by1': args['add1by1'],
                           'clsfy_loss': args['clsfy_loss'], 'gumbel_read': args['gumbel_read']}
        else:
            memory_init = None

        network_init = {'in_ch': 3, 'nclass': 19, 'backbone': args['backbone'], 'output_stride': 8,
                        'pretrained': True,'hideandseek': args['hideandseek'],'memory_init': memory_init}


    else:
        assert False, 'There is no network model named :' + args['network']

    args.update({'network_init': network_init})

    pprint(args)

    return args

def train():
    args = init()
    assert len(args['target'])
    framework = MetaFrameWork(args)
    framework.do_train()

def draw_tsne():
    args = init()
    framework = MetaFrameWork(args)
    # framework.draw_tsne_domcls(perplexities=[30], learning_rate=10, imagenum=8,pthname = 'best_city',all_class = True)
    framework.draw_mean_tsne_domcls(perplexities=[30], learning_rate=10, imagenum=200,pthname = 'best_city',all_class = True)


def predict():
    args = init()
    framework = MetaFrameWork(args)
    framework.predict_target(load_path='best_city', output_path='predictions',savenum = 3,inputimgname='munster_000065_000019_leftImg8bit.png')

def eval():
    args = init()
    name = args['name']
    targets = "CBMGS" # ex: GSC
    framework = MetaFrameWork(args)
    for target in targets:
        # for one experiment, test multi targets in multi batch_sizes
        framework.print('=' * 20 + ' {} '.format(name) + '=' * 20)
        framework.save_path = Path('experiment/'+name)

        framework.load('best_city', strict=False)
        # framework.load('ckpt', strict=False)

        seeds = [123456]
        for i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            framework.log('-' * 20 + ' {} '.format(target) + '-' * 20 + 'seed : '+str(seed) + '\n\n')

            if target == 'M':
                target_loader = get_target_loader(target, args['test_size'], mode='test',shuffle=False)
                target_loader.dataset.transforms = FixScale(short_size = 1000) # for gpu memory
            else:
                target_loader = get_target_loader(target, args['test_size'], mode='test', shuffle=False)

            framework.val(target_loader)

if __name__ == '__main__':
    # from utils.task import FunctionJob
    # job = FunctionJob([train], gpus=[[1]])
    # job.run(minimum_memory=10000)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # train()
    # draw_tsne()
    # predict()
    eval()
    # validation 모드는 center crop transform 되며(이미지 사이즈가 데이터셋에서 다를수 있기 때문) test 모드는 그렇지 않음
    # cityscapse는 test label이 없기 때문에 natural dataset 모듈에서 test가 validation으로 output 된다.(split ...함수에서 output의 test부분을 dev로 바꾸면 됨)