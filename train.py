from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

import models, loader, optim
import numpy as np
import pickle
from utils import *

# parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
# ap = parser.add_argument
# ap('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
# ap('-b',help='Batch size', type=int, default=128)
# ap('-B', help='Max epochs', type=int, default=100)
# ap('--lr', help='Learning rate', type=float, default=0.1)
# ap('--l2', help='L2', type=float, default=0.0)
# ap('-L', help='Langevin iterations', type=int, default=0)
# ap('--gamma', help='gamma', type=float, default=1e-4)
# ap('--scoping', help='scoping', type=float, default=1e-3)
# ap('--noise', help='SGLD noise', type=float, default=1e-4)
# ap('-g', help='GPU idx.', type=int, default=0)
# ap('-s', help='seed', type=int, default=42)
#opt = vars(parser.parse_args())
opt = {'m': 'cmapssfc', 'b': 128, 'B': 1, 'lr': 1., 'l2': 0.0, 'L': 20, 'gamma': 0.0001, 'scoping': 0.001, 'noise': 0.0001, 'g': 0, 's': 42}

#dataset_list = ['FD001','FD002','FD003','FD004']
dataset_list = ['FD004']
model_list = ['cmapsscnn']
algorithm_list = ['Adam']
#algorithm = 'MCMC_SDCA'
for algorithm in algorithm_list:
    for i_model in model_list:
        for i_dataset in dataset_list:
            if algorithm in ['Adam', 'Adagrad', 'RMSprop']:
                epoch_scale = 20
            else:
                epoch_scale = 1
                
            if i_model == 'cmapssfc':
                epochs = 5*epoch_scale
            else:
                epochs = 1*epoch_scale
                    
            opt = {'m': i_model, 'b': 128, 'B': epochs, 'lr': 1., 'l2': 0.0, 'L': 20, 'gamma': 0.0001, 'scoping': 0.001, 'noise': 0.0001, 'g': 0, 's': 42}
    
            th.set_num_threads(2)
            opt['cuda'] = th.cuda.is_available()
            if opt['cuda']:
                opt['g'] = -1
                th.cuda.set_device(opt['g'])
                th.cuda.manual_seed(opt['s'])
                cudnn.benchmark = True
            random.seed(opt['s'])
            np.random.seed(opt['s'])
            th.manual_seed(opt['s'])
    
            if 'mnist' in opt['m']:
                opt['dataset'] = 'mnist'
            elif 'allcnn' in opt['m']:
                opt['dataset'] = 'cifar10'
            elif 'cmapss' in opt['m']:
                opt['dataset'] = 'cmapss'
            else:
                assert False, "Unknown opt['m']: " + opt['m']
    
    
            if opt['dataset'] == 'cmapss':
                remain_sens = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
                if 'fc' in opt['m']:
                    style = 'FNN'
                elif 'cnn' in opt['m']:
                    style = 'CNN'
                dataset = i_dataset
                attr = dict(dataset=dataset, remain_sens=remain_sens, sqn_len=10, exp_smth=0.01, style=style)
                train_loader, val_loader, test_loader, input_shape = getattr(loader, opt['dataset'])(opt,attr)
            else:
                train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt)
    
            if 'cmapss' in opt['m']:
                model = getattr(models, opt['m'])(opt, input_shape=input_shape)
                print(model)
            else:   
                model = getattr(models, opt['m'])(opt)
    
            if opt['dataset'] == 'cmapss':
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()
    
            if opt['cuda']:
                model = model.cuda()
                criterion = criterion.cuda()
            
            # is_Adam = False
            # optimizer = optim.EntropySGD(model.parameters(),
            #         config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
            #         L=opt['L'], eps=opt['noise'], g0=opt['gamma'], g1=opt['scoping']), sdca=True)
    
    #         optimizer = optim.MCMC_SDCA(model.parameters(),
    #                 config = dict(lr=opt['lr'], momentum=0.0, nesterov=True, weight_decay=opt['l2'],
    #                 L=opt['L'], eps=opt['noise'], g0=opt['gamma'], g1=opt['scoping']))
            
            is_benchmark = True
            if algorithm == 'Adagrad':
                optimizer = th.optim.Adagrad(model.parameters())
            elif algorithm == 'RMSprop':
                optimizer = th.optim.RMSprop(model.parameters())
            elif algorithm == 'Adam':
                optimizer = th.optim.RMSprop(model.parameters())
                
            print(opt)
    
            def train(e):
                model.train()
    
                fs, top1 = AverageMeter(), AverageMeter()
                ts = timer()
    
                fs_list = list()
    
                bsz = opt['b']
                maxb = int(math.ceil(train_loader.n/bsz))
    
                for bi in range(maxb):
                    def helper():
                        def feval():
                            x,y = next(train_loader)
                            y = th.unsqueeze(y,1)
                            if opt['cuda']:
                                x,y = x.cuda(), y.cuda()
    
                            if opt['dataset'] == 'cmapss':
                                x, y = Variable(x), Variable(y)
                            else:
                                x, y = Variable(x), Variable(y.squeeze())
                            bsz = x.size(0)
    
                            optimizer.zero_grad()
                            yh = model(x)
                            f = criterion.forward(yh, y)
                            f.backward()
                            if opt['dataset'] == 'cmapss':
                                return f.data
                            else:
                                prec1, = accuracy(yh.data, y.data, topk=(1,))
                                err = 100.-prec1
                                return (f.data, err)
                        return feval
    
                    if opt['dataset'] == 'cmapss':
                        if not is_benchmark:
                            f = optimizer.step(helper(), model, criterion)
                        else:
                            feval = helper()
                            f = feval()
                            optimizer.step()
                             
                        if is_benchmark:
                            fs.update(f, bsz)
                            if bi%opt['L'] == 0:
                                fs_list.append(fs.avg)
                        else:
                            fs.update(f, bsz)
                            fs_list.append(fs.avg)
                    else:
                        f, err = optimizer.step(helper(), model, criterion)
                        fs.update(f, bsz)
                        top1.update(err, bsz)
    
                    if bi % 1 == 0 and bi != 0:  # bi%100
                        if opt['dataset'] == 'cmapss':
                            print('[%2d][%4d/%4d] %2.4f'%(e,bi,maxb, fs.avg))
                        else:
                            print('[%2d][%4d/%4d] %2.4f %2.2f%%'%(e,bi,maxb, fs.avg, top1.avg))
    
                if opt['dataset'] == 'cmapss':
                    print('Train: [%2d] %2.4f [%.2fs]'% (e, fs.avg, timer()-ts))
                else:
                    print('Train: [%2d] %2.4f %2.2f%% [%.2fs]'% (e, fs.avg, top1.avg, timer()-ts))
    
                return fs_list, timer()-ts
    
    
            def set_dropout(cache = None, p=0):
                if cache is None:
                    cache = []
                    for l in model.modules():
                        if 'Dropout' in str(type(l)):
                            cache.append(l.p)
                            l.p = p
                    return cache
                else:
                    for l in model.modules():
                        if 'Dropout' in str(type(l)):
                            assert len(cache) > 0, 'cache is empty'
                            l.p = cache.pop(0)
    
            def dry_feed():
                cache = set_dropout()
                maxb = int(math.ceil(train_loader.n/opt['b']))
                for bi in range(maxb):
                    x,y = next(train_loader)
                    if opt['cuda']:
                        x,y = x.cuda(), y.cuda()
                    x,y =   Variable(x, volatile=True), \
                            Variable(y.squeeze(), volatile=True)
                    yh = model(x)
                set_dropout(cache)
    
            def val(e, data_loader, test=False):
                dry_feed()
                model.eval()
    
                maxb = int(math.ceil(data_loader.n/opt['b']))
    
                fs, top1 = AverageMeter(), AverageMeter()
                for bi in range(maxb):
                    x,y = next(data_loader)
                    y = th.unsqueeze(y,1)
                    bsz = x.size(0)
    
                    if opt['cuda']:
                        x,y = x.cuda(), y.cuda()
    
                    if opt['dataset'] == 'cmapss':
                        x, y = Variable(x, volatile=True), Variable(y, volatile=True)
                    else:
                        x,y = Variable(x, volatile=True), Variable(y.squeeze(), volatile=True)
    
                    yh = model(x)
    
                    f = criterion.forward(yh, y).data
                    fs.update(f, bsz)
                    if not (opt['dataset'] == 'cmapss'):
                        prec1, = accuracy(yh.data, y.data, topk=(1,))
                        err = 100-prec1
                        top1.update(err, bsz)
                if not test:
                    if opt['dataset'] == 'cmapss':
                        print('Validation: [%2d] %2.4f\n'%(e, fs.avg))
                    else:
                        print('Validation: [%2d] %2.4f %2.4f%%\n'%(e, fs.avg, top1.avg))
                else:
                    if opt['dataset'] == 'cmapss':
                        print('Test: [%2d] %2.4f\n'%(e, fs.avg))
                    else:
                        print('Test: [%2d] %2.4f %2.4f%%\n'%(e, fs.avg, top1.avg))
                return fs.avg
    
    
            fs_list_total = list()
            time_total = 0
            for e in range(opt['B']):
                fs_list_e, time_e = train(e)
                fs_list_total = fs_list_total + fs_list_e
                time_total = time_total + time_e
                if e == opt['B']-1:
                    fs_val = val(e, val_loader)
                    fs_test = val(e, test_loader, test=True)
    
            with open(opt['m']+dataset+algorithm+'.pickle','wb') as f:
                pickle.dump([fs_list_total, time_total, fs_val, fs_test],f)
    
            with open(opt['m']+dataset+algorithm+'.pickle','rb') as f:
                [fs_list_total, time_total, fs_val, fs_test] = pickle.load(f)

        # plt.plot(fs_list_total)
        # plt.show()


# # plot figures:
# model_list = ['cmapsscnn']
# dataset_list = ['FD001','FD002','FD003']
# for i_model in model_list:
#     for i_dataset in dataset_list:

#         with open(i_model+i_dataset+'MCMCSDCA'+'.pickle','rb') as f:
#             [fs_list_total1, time_total1, fs_val1, fs_test1] = pickle.load(f)
#         with open(i_model+i_dataset+'Adam'+'.pickle','rb') as g:
#             [fs_list_total2, time_total2, fs_val2, fs_test2] = pickle.load(g)
#         with open(i_model+i_dataset+'Adagrad'+'.pickle','rb') as g:
#             [fs_list_total3, time_total3, fs_val3, fs_test3] = pickle.load(g)
#         with open(i_model+i_dataset+'RMSprop'+'.pickle','rb') as g:
#             [fs_list_total4, time_total4, fs_val4, fs_test4] = pickle.load(g)
            
#         print(i_model+i_dataset)
#         print(np.sqrt(fs_list_total3[-1].numpy()),'/',np.sqrt(fs_val3).numpy(),'/',np.sqrt(fs_test3).numpy(),'/',time_total3)
        
#         plt.figure()
#         plt.plot(fs_list_total1, label='MCMC_SDCA')
#         plt.plot(fs_list_total2, label='Adam', linestyle='dashed')
#         plt.plot(fs_list_total3, label='Adagrad')
#         plt.plot(fs_list_total4, label='RMSprop', linestyle='dashed')
#         plt.legend()
#         plt.xlabel('#grads/Langevin')
#         plt.ylabel('train mse')
#         plt.savefig(i_model+i_dataset+'.png')
#        plt.close()
        
        