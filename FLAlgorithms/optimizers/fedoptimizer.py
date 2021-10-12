


from torch.optim import Optimizer
import torch
import math
from torch import Tensor
from typing import List, Optional


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                # p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
                # p.data = p.data - group['lr'] * (p.grad.data + 0.0001 * (p.data - localweight.data) + 0.0001*p.data)
                p.data = p.data - group['lr'] * p.grad.data
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']


    


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, beta = 1, n_k = 1):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta  * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss

class pFedMeAdamOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0, mu=0, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, lamda=lamda, mu=mu, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(pFedMeAdamOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pFedMeAdamOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss=None
        if closure is not None:
            loss = closure
        # weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #print('grad: {}\t'.format(grad))
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]   # 之前的step累计数据

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data) # [batch, seq]
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] # 上次的r与s
                if amsgrad:
                # asmgrad优化方法是针对Adam的改进，通过添加额外的约束，使学习率始终为正值。
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # 序号对应最后一幅图中序号
                if group['weight_decay'] != 0:  # 进行权重衰减(实际是L2正则化）
                	# 6. grad(t)=grad(t-1)+ weight*p(t-1)
                    grad.add_(group['weight_decay'], p.data)

                # personalization
                # if group['lamda'] != 0:
                    # print('personalization!')
                    # print("grad before before:{}".format(grad.data))
                    # grad.add_(p.data - localweight.data, alpha=0.01*group['lamda'])
                    # print("grad before:{}".format(grad.data))
                    #grad.add_(p.data, alpha=group['mu'])
                    # print("grad after:{}".format(grad.data))

                # Decay the first and second moment running average coefficient
                # 7.计算m(t): m(t)=beta_1*m(t-1)+(1-beta_1)*grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 8.计算v(t): v(t)= beta_2*v(t-1)+(1-beta_2)*grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    # 迭代改变max_exp_avg_sq的值（取最大值），传到下一次，保留之前的梯度信息。
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                	# 计算sqrt(v(t))+epsilon
                	# sqrt(v(t))+eps = denom = sqrt(v(t))/sqrt(1-beta_2^t)+eps
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
				# step_size=lr/bias_correction1=lr/(1-beta_1^t)
                step_size = group['lr'] / bias_correction1
				#p(t)=p(t-1)-step_size*m(t)/denom
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        return group['params']



