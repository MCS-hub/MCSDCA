import torch as th
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import os, sys, pdb
from cmapss_process import preprocess

class sampler_t:
    def __init__(self, batch_size, x,y, train=True):
        self.n = x.size(0)
        self.x, self.y = x,y
        self.b = batch_size
        self.idx = th.arange(0, self.b).long()
        self.train = train
        self.sidx = 0

    def __next__(self):
        if self.train:
            self.idx.random_(0,self.n)
        else:
            s = self.sidx
            e = min(s+self.b-1, self.n)
            #print s,e

            self.idx = th.arange(s, e).long()
            self.sidx += self.b
            if self.sidx >= self.n:
                self.sidx = 0

        x,y  = th.index_select(self.x, 0, self.idx), \
            th.index_select(self.y, 0, self.idx)
        return x, y

    next = __next__

    def __iter__(self):
        return self

def mnist(opt):
    d1, d2 = datasets.MNIST('../proc', download=True, train=True), \
            datasets.MNIST('../proc', train=False)

    train = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels)
    val = sampler_t(opt['b'], d2.test_data.view(-1,1,28,28).float(),
        d2.test_labels, train=False)
    return train, val, val

def cifar10(opt):
    loc = '../proc/'
    d1 = np.load(loc+'cifar10-train.npz')
    d2 = np.load(loc+'cifar10-test.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']))
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val

def cmapss(opt, attr=None):
    assert attr is not None, 'attach attributes to CMAPSS dataset'
    dataset = attr['dataset']
    remaining_sensors = attr['remain_sens']
    sequence_length = attr['sqn_len']
    exp_smooth = attr['exp_smth']
    style = attr['style']
    seed = attr['seed']
    
    train_array, train_label, test_array, test_label, train_split_array, \
    train_split_label, val_split_array, val_split_label \
    = preprocess(dataset=dataset, remaining_sensors=remaining_sensors, sequence_length=sequence_length, exp_smooth=exp_smooth, style=style, return_type='numpy', seed=seed)

    train = sampler_t(opt['b'], th.from_numpy(train_split_array), th.from_numpy(train_split_label))
    val = sampler_t(opt['b'], th.from_numpy(val_split_array), th.from_numpy(val_split_label), train=False)
    test = sampler_t(opt['b'], th.from_numpy(test_array), th.from_numpy(test_label), train=False)
            
    return train, val, test
