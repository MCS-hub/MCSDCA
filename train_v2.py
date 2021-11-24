from __future__ import print_function
import argparse, math, random
import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from copy import deepcopy

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
# opt = vars(parser.parse_args())
random_seed_list = [40, 41, 42, 43, 44]

torch.manual_seed(42)
# check gpu
cuda_available = False
if torch.cuda.is_available():
    cuda_available = True

mc_FD001_FNN = {'activation': 'Sigmoid', 'dropout': 0.4, 'structure': [128, 256, 512], 'input_sz': 294, 'cuda': cuda_available}
sq_FD001_FNN = 21
exps_FD001_FNN = 0.05
FD001_FNN = {'mc': mc_FD001_FNN, 'sq': sq_FD001_FNN, 'exps': exps_FD001_FNN}

mc_FD001_LSTM = {'dropout': 0.3, 'num_layers': 1, 'input_sz': 14, 'hidden_size': 128, 'cuda': cuda_available}
sq_FD001_LSTM = 23
exps_FD001_LSTM = 0.1
FD001_LSTM = {'mc': mc_FD001_LSTM, 'sq': sq_FD001_LSTM, 'exps': exps_FD001_LSTM}

mc_FD002_FNN = {'activation': 'Sigmoid', 'dropout': 0.6, 'structure': [64, 128, 256], 'input_sz': 266, 'cuda': cuda_available}
sq_FD002_FNN = 19
exps_FD002_FNN = 0.05
FD002_FNN = {'mc': mc_FD002_FNN, 'sq': sq_FD002_FNN, 'exps': exps_FD002_FNN}

mc_FD002_LSTM = {'dropout': 0.5, 'num_layers': 1, 'input_sz': 14, 'hidden_size': 256, 'cuda': cuda_available}
sq_FD002_LSTM = 19
exps_FD002_LSTM = 0.05
FD002_LSTM = {'mc': mc_FD002_LSTM, 'sq': sq_FD002_LSTM, 'exps': exps_FD002_LSTM}


mc_FD003_FNN = {'activation': 'Tanh', 'dropout': 0.1, 'structure': [64, 128, 256], 'input_sz': 476, 'cuda': cuda_available}
sq_FD003_FNN = 34
exps_FD003_FNN = 0.5
FD003_FNN = {'mc': mc_FD003_FNN, 'sq': sq_FD003_FNN, 'exps': exps_FD003_FNN}

mc_FD003_LSTM = {'dropout': 0.0, 'num_layers': 1, 'input_sz': 14, 'hidden_size': 128, 'cuda': cuda_available}
sq_FD003_LSTM = 34
exps_FD003_LSTM = 0.3
FD003_LSTM = {'mc': mc_FD003_LSTM, 'sq': sq_FD003_LSTM, 'exps': exps_FD003_LSTM}


mc_FD004_FNN = {'activation': 'Sigmoid', 'dropout': 0.6, 'structure': [64, 128, 256], 'input_sz': 154, 'cuda': cuda_available}
sq_FD004_FNN = 11
exps_FD004_FNN = 0.05
FD004_FNN = {'mc': mc_FD004_FNN, 'sq': sq_FD004_FNN, 'exps': exps_FD004_FNN}

mc_FD004_LSTM = {'dropout': 0.0, 'num_layers': 1, 'input_sz': 14, 'hidden_size': 64, 'cuda': cuda_available}
sq_FD004_LSTM = 18
exps_FD004_LSTM = 0.01
FD004_LSTM = {'mc': mc_FD004_LSTM, 'sq': sq_FD004_LSTM, 'exps': exps_FD004_LSTM}

ref_dict = {'FD001_FNN': FD001_FNN, 'FD001_LSTM': FD001_LSTM, 'FD002_FNN': FD002_FNN, 'FD002_LSTM': FD002_LSTM,
            'FD003_FNN': FD003_FNN, 'FD003_LSTM': FD003_LSTM, 'FD004_FNN': FD004_FNN, 'FD004_LSTM': FD004_LSTM}


dataset_list = ['FD001','FD002','FD003','FD004']

model_list = ['cmapssfnn', 'cmapsslstm']

algorithm_list = ['EntropySDCA', 'Adam', 'Adagrad', 'RMSprop']
#algorithm_list = ['EntropySDCA', 'Adagrad']

remain_sens = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17','s_20', 's_21']
is_benchmark = False
criterion = nn.MSELoss()

for i_dataset in dataset_list:
    for i_model in model_list:
        # get model config info and data set preprocessing info
        model_type = None
        if 'fnn' in i_model:
            model_type = 'FNN'
        elif 'lstm' in i_model:
            model_type = 'LSTM'

        model_data_info = ref_dict[i_dataset + '_' + model_type]
        sequence_length = model_data_info['sq']
        model_config = model_data_info['mc']
        exp_smooth = model_data_info['exps']

        opt = {'m': i_model, 'b': 32, 'lr': 0.01, 'l2': 0.0, 'L': 20, 'gamma': 0.0001, 'scoping': 0.001,
               'noise': 0.0001, 'g': 0, 's': 42, 'dataset': 'cmapss'}
        opt['cuda'] = torch.cuda.is_available()

        if 'fnn' in opt['m']:
            style = 'FNN'
        elif 'cnn' in opt['m']:
            style = 'CNN'
        else:
            style = 'LSTM'

        for seed in random_seed_list:
            attr = dict(dataset=i_dataset, remain_sens=remain_sens, sqn_len=sequence_length, exp_smth=exp_smooth,
                        style=style, seed=seed)
            train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt, attr)
            model = getattr(models, opt['m'])(model_config)
            print(model)
            if opt['cuda']:
                model = model.cuda()
                criterion = criterion.cuda()

            initial_w = deepcopy(model.state_dict())

            for algorithm in algorithm_list:
                # set the model initial points
                with torch.no_grad():
                    for param_tensor in initial_w:
                        model.state_dict()[param_tensor].copy_(initial_w[param_tensor])

                if algorithm in ['Adam', 'Adagrad', 'RMSprop']:
                    epoch_scale = 20
                else:
                    epoch_scale = 1
                epochs = 2*epoch_scale

                opt['B'] = epochs

                torch.set_num_threads(2)
                random.seed(opt['s'])
                np.random.seed(opt['s'])
                torch.manual_seed(opt['s'])


                if algorithm == 'EntropySDCA':
                    is_benchmark = False
                    optimizer = optim.EntropySGD(model.parameters(),config=dict(lr=opt['lr'], momentum=0.0, nesterov=False,
                                                             weight_decay=opt['l2'],
                                                             L=opt['L'], eps=opt['noise'], g0=opt['gamma'],
                                                             g1=opt['scoping']), sdca=True, proximal=1e-5)
                elif algorithm == 'MCMC_SDCA':
                    is_benchmark = False
                    print('oh really')
                    import time
                    time.sleep(10)
                    optimizer = optim.MCMC_SDCA(model.parameters(),config=dict(lr=opt['lr'], momentum=0.0, nesterov=True, weight_decay=opt['l2'],
                                                        L=opt['L'], eps=opt['noise'], g0=opt['gamma'], g1=opt['scoping']))

                else:
                    is_benchmark = True
                    if algorithm == 'Adagrad':
                        optimizer = torch.optim.Adagrad(model.parameters())
                    elif algorithm == 'RMSprop':
                        optimizer = torch.optim.RMSprop(model.parameters())
                    elif algorithm == 'Adam':
                        optimizer = torch.optim.Adam(model.parameters())

                print(opt)


                def train(e):
                    model.train()

                    fs, top1 = AverageMeter(), AverageMeter()
                    ts = timer()

                    fs_list = list()

                    bsz = opt['b']
                    maxb = int(math.ceil(train_loader.n / bsz))

                    for bi in range(maxb):
                        def helper():
                            def feval():
                                x, y = next(train_loader)
                                y = torch.unsqueeze(y, 1)
                                if opt['cuda']:
                                    x, y = x.cuda(), y.cuda()
                                x, y = Variable(x), Variable(y)
                                bsz = x.size(0)
                                optimizer.zero_grad()
                                yh = model(x)
                                f = criterion.forward(yh, y)
                                f.backward()
                                return f.data
                            return feval

                        if not is_benchmark:
                            f = optimizer.step(helper(), model, criterion)
                        else:
                            feval = helper()
                            f = feval()
                            optimizer.step()

                        if is_benchmark:
                            fs.update(f, bsz)
                            if bi % opt['L'] == 0:
                                fs_list.append(fs.avg)
                        else:
                            fs.update(f, bsz)
                            fs_list.append(fs.avg)


                        if bi % 1 == 0 and bi != 0:  # bi%100
                            print('[%2d][%4d/%4d] %2.4f' % (e, bi, maxb, fs.avg))

                    print('Train: [%2d] %2.4f [%.2fs]' % (e, fs.avg, timer() - ts))

                    return fs_list, timer() - ts


                def set_dropout(cache=None, p=0):
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
                    maxb = int(math.ceil(train_loader.n / opt['b']))
                    for bi in range(maxb):
                        x, y = next(train_loader)
                        if opt['cuda']:
                            x, y = x.cuda(), y.cuda()
                        x, y = Variable(x, volatile=True), \
                               Variable(y.squeeze(), volatile=True)
                        yh = model(x)
                    set_dropout(cache)


                def val(e, data_loader, test=False, draw_predict = False, fig_name = None):
                    dry_feed()
                    model.eval()

                    maxb = int(math.ceil(data_loader.n / opt['b']))

                    fs, top1 = AverageMeter(), AverageMeter()

                    # for draw prediction and truth
                    if test:
                        predict_list = list()
                        truth_list = list()

                    for bi in range(maxb):
                        x, y = next(data_loader)
                        if test:
                            truth_list = truth_list + y.tolist()
                        y = torch.unsqueeze(y, 1)
                        bsz = x.size(0)

                        if opt['cuda']:
                            x, y = x.cuda(), y.cuda()

                        x, y = Variable(x, volatile=True), Variable(y, volatile=True)

                        yh = model(x)

                        if test:
                            predict_list = predict_list + yh.squeeze().tolist()

                        f = criterion.forward(yh, y).data
                        fs.update(f, bsz)

                    if not test:
                        print('Validation rmse: [%2d] %2.4f\n' % (e, np.sqrt(fs.avg)))

                    else:
                        print('Test rmse: [%2d] %2.4f\n' % (e, np.sqrt(fs.avg)))

                    if draw_predict and test:
                        plt.figure()
                        plt.plot(truth_list, label='True')
                        plt.plot(predict_list, label='Prediction', linestyle='dashed')
                        plt.legend()
                        plt.xlabel('Engine')
                        plt.ylabel('Remaining Useful Life')
                        plt.savefig(fig_name + '.png')
                        plt.close()

                    return fs.avg

                fs_list_total = list()
                time_total = 0
                for e in range(opt['B']):
                    fs_list_e, time_e = train(e)
                    fs_list_total = fs_list_total + fs_list_e
                    time_total = time_total + time_e
                    if e == opt['B'] - 1:
                        if algorithm == 'EntropySDCA':
                            fig_name = opt['m'] + i_dataset + algorithm + 'prediction'+'_seed'+str(seed)
                            fs_val = val(e, val_loader)
                            fs_test = val(e, test_loader, test=True, draw_predict = True, fig_name = fig_name)
                        else:
                            fs_val = val(e, val_loader)
                            fs_test = val(e, test_loader, test=True)

                with open(opt['m'] + i_dataset + algorithm + '_seed' + str(seed)+'.pickle', 'wb') as f:
                    pickle.dump([fs_list_total, time_total, fs_val, fs_test], f)

                with open(opt['m'] + i_dataset + algorithm + '_seed' + str(seed) + '.pickle', 'rb') as f:
                    [fs_list_total, time_total, fs_val, fs_test] = pickle.load(f)

        # plt.plot(fs_list_total)
        # plt.show()

# # plot figures:
# import pickle
# from matplotlib import pyplot as plt
# model_list = ['cmapssfnn']
# dataset_list = ['FD001']
# for i_model in model_list:
#     for i_dataset in dataset_list:
#
#         with open(i_model+i_dataset+'EntropySDCA'+'.pickle','rb') as f:
#             [fs_list_total1, time_total1, fs_val1, fs_test1] = pickle.load(f)
#         with open(i_model+i_dataset+'Adam'+'.pickle','rb') as g:
#             [fs_list_total2, time_total2, fs_val2, fs_test2] = pickle.load(g)
#         with open(i_model+i_dataset+'Adagrad'+'.pickle','rb') as g:
#             [fs_list_total3, time_total3, fs_val3, fs_test3] = pickle.load(g)
#         with open(i_model+i_dataset+'RMSprop'+'.pickle','rb') as g:
#             [fs_list_total4, time_total4, fs_val4, fs_test4] = pickle.load(g)
#
#         # print(i_model+i_dataset)
#         # print(np.sqrt(fs_list_total3[-1].numpy()),'/',np.sqrt(fs_val3).numpy(),'/',np.sqrt(fs_test3).numpy(),'/',time_total3)
#         print(np.sqrt(fs_list_total1[-1].numpy()), '/', np.sqrt(fs_val1).numpy(), '/', np.sqrt(fs_test1).numpy(), '/',time_total1)
#         print(np.sqrt(fs_list_total2[-1].numpy()), '/', np.sqrt(fs_val2).numpy(), '/', np.sqrt(fs_test2).numpy(), '/',time_total2)
#         print(np.sqrt(fs_list_total3[-1].numpy()), '/', np.sqrt(fs_val3).numpy(), '/', np.sqrt(fs_test3).numpy(), '/',time_total3)
#         print(np.sqrt(fs_list_total4[-1].numpy()), '/', np.sqrt(fs_val4).numpy(), '/', np.sqrt(fs_test4).numpy(), '/',time_total4)
#
#         plt.figure()
#         plt.plot(fs_list_total1, label='EntropySDCA')
#         plt.plot(fs_list_total2, label='Adam', linestyle='dashed')
#         plt.plot(fs_list_total3, label='Adagrad')
#         plt.plot(fs_list_total4, label='RMSprop', linestyle='dashed')
#         plt.legend()
#         plt.xlabel('#grads/(minibatch*Langevin)')
#         plt.ylabel('train mse')
#         plt.savefig(i_model+i_dataset+'.png')
#         plt.close()



