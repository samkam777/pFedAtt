import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
import numpy as np
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer, pFedMeAdamOptimizer
from FLAlgorithms.users.userbase import User
import torch.optim as optim

from m_opacus.dp_model_inspector import DPModelInspector
from m_opacus.utils import module_modification
from m_opacus import PrivacyEngine


class UserpFedMe(User):
    def __init__(self, device, args, model, train_data, test_data, train_data_samples, numeric_id, running_time, hyper_param):
        super().__init__(device, args, model, train_data, test_data, train_data_samples, numeric_id, running_time, hyper_param)

        self.loss = nn.BCELoss(reduction=self.loss_reduction).to(device)
        self.K = args.K
        self.personal_learning_rate = args.personal_learning_rate
        self.optimizer = pFedMeAdamOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=0)

        if self.if_DP:
            self.privacy_engine = PrivacyEngine(
                self.model,
                #batch_size=self.virtual_batch_size,
                #batch_size=self.batch_size*0.01,    # 当时做二次抽样的时候用的
                batch_size=self.batch_size,
                sample_size=len(self.trainloader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                loss_reduction="mean",
            )
            self.privacy_engine.to(device)
            self.privacy_engine.attach(self.optimizer)
        #     print("use DP!")
        # else:
        #     print("Not use DP!")

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        losses = []
        self.model.train()
        self.optimizer.zero_grad()
        #real_times = self.virtual_batch_size / self.batch_size     # virtual step setting parameter
        #print("real_times:{}\t".format(real_times))
        for epoch in range(1, self.local_epochs + 1):  # local update
            
            user, item, label = self.get_next_train_batch()  
            #for index, (user, item, label) in enumerate(self.trainloader):

            ####### 二次采样部分 #######
            #if self._subsample:
            #    print("subsample data!!")
            #    indices = np.random.permutation(len(user))  # 打乱数据的随机数
            #    rnd_sampled = np.random.binomial(len(user), 0.01)   # 二项分布，用于二次采样数据
            #    user = user[indices][:rnd_sampled]
            #    item = item[indices][:rnd_sampled]
            #    label = label[indices][:rnd_sampled]
            ###########################

            user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)

            # K is number of personalized steps
            for i in range(self.K):  
                output = self.model(user, item)

                reg_loss = torch.tensor(0.).to(self.device)
                for p, local_p in zip(self.model.parameters(), self.p_local_model.parameters()):
                    reg_loss += (torch.norm(p-local_p, p=2)) ** 2

                loss = self.loss(output, label)
                pfedloss = loss + self.lamda*reg_loss
                pfedloss.backward()


                if self.if_DP:
                # 只有最后一个epoch加噪声
                    if(epoch == self.local_epochs):
                        print("add noise!")
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.original_step()
                        self.optimizer.zero_grad()

                    '''
                    # virtual step setting  在内存资源不足时，可以用来添加虚拟的更新过程，增大batchsize
                    if (epoch % real_times) == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.virtual_step()
                    '''
                else:
                    self.optimizer.step(self.p_local_model)
                    self.optimizer.zero_grad()

                # loss
                losses.append(pfedloss.item())

            for new_param, localweight in zip(self.model.parameters(), self.p_local_model.parameters()):
                localweight.data = localweight.data - self.learning_rate * (localweight.data - new_param.data)  

        #update local model as local_weight_upated
        self.update_parameters(self.p_local_model.parameters())

        # calculate loss
        train_loss = sum(losses) / len(losses)

        # evaluate
        self.user_persionalized_evaluate(epochs, train_loss)
        
        # calculate privacy budget
        if self.if_DP:
            self.privacy_engine.steps = epochs+1
            epsilons, _ = self.privacy_engine.get_privacy_spent(self.delta)
            print("training epochs {}  client {}  epsilons:{:.4f}\t".format(epochs, self.id, epsilons))
            return train_loss, epsilons
        else:
            return train_loss


        


