import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.5

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

class mnistconv(nn.Module):
    def __init__(self, opt):
        super(mnistconv, self).__init__()
        self.name = 'mnistconv'
        opt['d'] = 0.5

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,opt['d']),
            convbn(20,50,5,2,opt['d']),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

class allcnn(nn.Module):
    def __init__(self, opt = {'d':0.5}, c1=96, c2= 192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'
        opt['d'] = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


# class cmapssfc(nn.Module):
#     def __init__(self, opt, input_shape):
#         super(cmapssfc, self).__init__()
#         self.name = 'cmapss'
#
#         c = 1024
#         opt['d'] = 0.5
#
#         self.m = nn.Sequential(
#             View(input_shape),
#             nn.Dropout(0.2),
#             nn.Linear(input_shape,c),
#             nn.BatchNorm1d(c),
#             nn.ReLU(True),
#             nn.Dropout(opt['d']),
#             nn.Linear(c,c),
#             nn.BatchNorm1d(c),
#             nn.ReLU(True),
#             nn.Dropout(opt['d']),
#             nn.Linear(c,1))
#
#         s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
#         print(s)
#
#     def forward(self, x):
#         return self.m(x)

# class cmapsscnn(nn.Module):
#     def __init__(self, opt = {'d':0.5}, c1=96, c2= 192, input_shape=None):
#         super(cmapsscnn, self).__init__()
#         self.name = 'cmapsscnn'
#         opt['d'] = 0.5
#
#         def convbn(ci,co,ksz,s=1,pz=0):
#             return nn.Sequential(
#                 nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
#                 nn.BatchNorm2d(co),
#                 nn.ReLU(True))
#
#         self.m = nn.Sequential(
#             nn.Dropout(0.2),
#             convbn(1,c1,3,1,1),
#             #convbn(c1,c1,3,1,1),
#             #convbn(c1,c1,3,2,1),
#             nn.Dropout(opt['d']),
#             convbn(c1,c2,3,1,1),
#             #convbn(c2,c2,3,1,1),
#             #convbn(c2,c2,3,2,1),
#             #nn.Dropout(opt['d']),
#             #convbn(c2,c2,3,1,1),
#             #convbn(c2,c2,3,1,1),
#             convbn(c2,10,1,1),
#             nn.AvgPool2d(8),
#             nn.Flatten(),
#             nn.Linear(10,1))
#
#         s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
#         print(s)
#
#     def forward(self, x):
#         return self.m(x)


        
class cmapssfnn(nn.Module):
    def __init__(self, mc):
        super(cmapssfnn, self).__init__()
        self.name = 'FNN'
        st = mc['structure']
        act_str = mc['activation']
        dropout = mc['dropout']
        self.m = nn.Sequential(
            nn.Linear(mc['input_sz'], st[0]),
            getattr(nn, act_str)(),
            nn.Dropout(dropout),
            nn.Linear(st[0], st[1]),
            getattr(nn, act_str)(),
            nn.Dropout(dropout),
            nn.Linear(st[1], st[2]),
            getattr(nn, act_str)(),
            nn.Dropout(dropout),
            nn.Linear(st[2], 1)
        )

    def forward(self, x):
        return self.m(x)


class cmapsslstm(nn.Module):

    def __init__(self, mc):
        super(cmapsslstm, self).__init__()

        self.num_layers = mc['num_layers']
        self.input_sz = mc['input_sz']
        self.hidden_sz = mc['hidden_size']
        self.cuda = mc['cuda']

        self.lstm = nn.LSTM(input_size=self.input_sz, hidden_size=self.hidden_sz,
                            num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(mc['dropout'])
        self.fc = nn.Linear(self.hidden_sz, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_sz))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_sz))
        if self.cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        out = out[:,-1,:]
        out = out.view(-1,self.hidden_sz)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class cmapssrnn(nn.Module):

    def __init__(self, mc):
        super(cmapssrnn, self).__init__()

        self.num_layers = mc['num_layers']
        self.input_sz = mc['input_sz']
        self.hidden_sz = mc['hidden_size']
        self.cuda = mc['cuda']

        self.rnn = nn.RNN(input_size=self.input_sz, hidden_size=self.hidden_sz,
                            num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(mc['dropout'])
        self.fc = nn.Linear(self.hidden_sz, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_sz))
        if self.cuda:
            h_0 = h_0.cuda()

        # Propagate input through RNN
        out, h_out = self.rnn(x, h_0)
        out = out[:,-1,:]
        out = out.view(-1,self.hidden_sz)
        out = self.dropout(out)
        out = self.fc(out)
        return out