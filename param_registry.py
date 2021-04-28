# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(model,source, target):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New model / networks / etc. should add entries here.
    """

    hparams = {'debug': False,
     'inner_lr': 0.001,
     'lamb_cpt': 0.1,
     'lamb_sep': 0.2,
     'mem_after_update': True,
     'name': 'exp_1e2_supmem_mldg_after_lrplate',
     'network': model,
     'network_init': {'backbone': 'resnet50',
                      'freeze_backbone': False,
                      'freeze_bn': False,
                      'in_ch': 3,
                      'memory_init': None,
                      'nclass': 19,
                      'output_stride': 8,
                      'pretrained': True,
                      'skipconnect': True},
     'no_inner_memloss': False,
     'no_outer_memloss': False,
     'outer_lr': 0.005,
     'resume': False,
     'sche': 'lrplate',
     'source': source,
     'target': target,
     'test_size': 2,
     'train_num': 1,
     'train_size': 2}

    if model in ['DANN', 'CDANN']:
        memory_init = {'feature_dim': 256,
                                      'key_dim': 256,
                                      'memory_size': 19,
                                      'momentum': 0.2,
                                      'supervised_mem': True,
                                      'temp_gather': 0.1,
                                      'temp_update': 0.1}
        hparams['network_init']

    elif model == "RSC":
        _hparam('lambda', 1.0)
        _hparam('lambda', 1.0)
        _hparam('lambda', 1.0)



    return hparams

def default_hparams(algorithm, source, target):
    return _hparams(algorithm, source, target)