from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
import torch as th

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}, sdca = False, proximal = 0.):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]
        # if sdca, set the lr = 1.
        if sdca:
            config['lr'] = 1./(1.+proximal)

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'      
        mfmerr = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        params = self.param_groups[0]['params']

        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'] = [], []
            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))

            state['langevin'] = dict(mw=deepcopy(state['wc']),
                                    mdw=deepcopy(state['mdw']),
                                    eta=deepcopy(state['mdw']),
                                    lr = 0.001, #0.001
                                    beta1 = 0.75)

        lp = state['langevin']
        for i,w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
        llr, beta1 = lp['lr'], lp['beta1']
        g = g0*(1+g1)**state['t']
        print('g: ',g)

        for i in range(L):
            #f,_ = closure()
            f = closure()
            for wc,w,mw,mdw,eta in zip(state['wc'], params, \
                                    lp['mw'], lp['mdw'], lp['eta']):
                dw = w.grad.data

                if wd > 0:
                    dw.add_(wd, w.data)
                if mom > 0:
                    mdw.mul_(mom).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom, mdw)
                    else:
                        dw = mdw

                # add noise
                eta.normal_()
                dw.add_(-g, wc-w.data).add_(eps/np.sqrt(0.5*llr), eta)

                # update weights
                w.data.add_(-llr, dw)
                #mw.mul_(beta1).add_(1-beta1, w.data)  #exponential
                mw.mul_((L+1)/(L+2)).add_(1/(L+2),w.data)  #normal average

        if L > 0:
            # copy model back
            for i,w in enumerate(params):
                w.data.copy_(state['wc'][i])
                w.grad.data.copy_(w.data-lp['mw'][i])

        for w,mdw,mw in zip(params, state['mdw'], lp['mw']):
            dw = w.grad.data

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw

            w.data.add_(-lr, dw)

        return mfmerr

    
class MCMC_SDCA(Optimizer):
    
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]
        
        # always set lr = 1.
        config['lr'] = 1.

        super(MCMC_SDCA, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        mfmerr = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']
        
        # haven't considered momemtum and weight_decay for a moment
        if wd>0. or mom>0.:
            raise NameError('not yet considered momemtum and weight_decay for a moment')

        params = self.param_groups[0]['params']

        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'], state['mmw'] = [], [], []  # add mean_mean_w to the optimizer state
            state['count'] = 0
            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))
                state['mmw'].append(0.*deepcopy(w.data))

            state['langevin'] = dict(mw=deepcopy(state['wc']),
                                    mdw=deepcopy(state['mdw']),
                                    eta=deepcopy(state['mdw']),
                                    lr = 0.001,
                                    beta1 = 0.75)
            
        #print(state['mmw'])
        lp = state['langevin']
        for i,w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
        llr, beta1 = lp['lr'], lp['beta1']
        g = g0*(1+g1)**state['t']

        for i in range(L):
            f = closure()
            for wc,w,mw,mdw,eta in zip(state['wc'], params, \
                                    lp['mw'], lp['mdw'], lp['eta']):
                dw = w.grad.data

                if wd > 0:
                    dw.add_(wd, w.data)
                if mom > 0:
                    mdw.mul_(mom).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom, mdw)
                    else:
                        dw = mdw

                # add noise
                eta.normal_()
                dw.add_(-g, wc-w.data).add_(eps/np.sqrt(0.5*llr), eta)

                # update weights
                w.data.add_(-llr, dw)
                mw.mul_(beta1).add_(1-beta1, w.data)

        if L > 0:
            # copy model back
            for i,w in enumerate(params):
                w.data.copy_(state['wc'][i])
                w.grad.data.copy_(w.data-lp['mw'][i])

        for w,mdw,mw,mmw in zip(params, state['mdw'], lp['mw'],state['mmw']):
            dw = w.grad.data

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw
        
            
            ct = state['count']
            #ct = 0.
            #w.data.add_(-lr,dw).mul_(1/(ct+1)).add_(ct/(ct+1),mmw)
            #print('hihi')
            alpha = 0.1
            w.data.add_(-lr,dw).mul_(1-alpha).add_(alpha,mmw)
            mmw.data.copy_(w.data)
            
        
        state['count'] += 1

        return mfmerr