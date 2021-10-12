import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from evaluate.evaluate import hit, ndcg
from logging_results import user_logging

class User:
    def __init__(self, device, args, model, train_data, test_data, train_data_samples, numeric_id, running_time, hyper_param):
        
        self.device = device
        self.model = copy.deepcopy(model)
        self.train_samples = train_data_samples
        self.id = numeric_id
        self.hyper_param = hyper_param
        self.running_time = running_time
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.beta = args.beta
        self.lamda = args.lamda
        self.trainloader = train_data
        self.testloader = test_data
        self.local_epochs = int(len(self.trainloader.dataset) / self.batch_size) 
        # self.local_epochs = args.local_epochs
        #print("local epochs:{}\t".format(self.local_epochs))
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        self.top_k = args.top_k

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))    
        self.p_local_model = copy.deepcopy(self.model)
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

        # DP
        self.delta = args.delta
        self.noise_multiplier = args.noise_multiplier
        self.max_grad_norm = args.max_grad_norm
        self.virtual_batch_size = args.virtual_batch_size
        self.if_DP = args.if_DP
        self._subsample = args._subsample

        self.loss_reduction = args.loss_reduction

############## get and set users' parameters ##############
    def set_parameters(self, model):
        for old_param, new_param, p_local_param, local_param in zip(self.model.parameters(), model.parameters(), self.p_local_model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            p_local_param.data = new_param.data.clone() # personal param update
            local_param.data = new_param.data.clone()   # average param update

    def set_grads(self, model):
        for old_param, new_param, p_local_param, local_param in zip(self.model.parameters(), model.parameters(), self.p_local_model.parameters(), self.local_model):
            old_param.grad.data = new_param.grad.data.clone()
            p_local_param.data = new_param.data.clone()
            local_param.grad.data = new_param.grad.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads
############################################################


###################### user evaluate ######################
    # FedAvg
    def test(self):
        self.model.eval()
        HR, NDCG = [], []
        for user, item, label in self.testloader:
            user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)
            predictions = self.model(user, item)
            _, indices = torch.topk(predictions, self.top_k)
            recommends = torch.take(item, indices).cpu().numpy().tolist()

            ng_item = item[0].item() # leave one-out evaluation has only one item per user
            HR.append(hit(ng_item, recommends))
            NDCG.append(ndcg(ng_item, recommends))
            # single client score
        HR_np = np.mean(HR)
        NDCG_np = np.mean(NDCG)
        
        return HR_np, NDCG_np

    # FedAvg
    def train_error_and_loss(self):
        self.model.eval()
        losses = [] 
        for user, item, label in self.trainloader:
            user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)
            prediction = self.model(user, item)

            loss = self.loss(prediction, label)
            losses.append(loss.item())

        return sum(losses) / len(losses)

    # pFedMe
    def test_persionalized_model(self):
        self.model.eval()
        #self.update_parameters(self.persionalized_model_bar)       # 恢复到训练前的，然后跑验证集
        HR, NDCG = [], []
        for user, item, label in self.testloader:
            user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)
            predictions = self.model(user, item)
            _, indices = torch.topk(predictions, self.top_k)
            recommends = torch.take(item, indices).cpu().numpy().tolist()

            ng_item = item[0].item() # leave one-out evaluation has only one item per user
            HR.append(hit(ng_item, recommends))
            NDCG.append(ndcg(ng_item, recommends))
        HR_np = np.mean(HR)
        NDCG_np = np.mean(NDCG)
        #self.update_parameters(self.local_model)                    # 跑完验证集后，再恢复到更新后的model
        return HR_np, NDCG_np

    # pFedMe
    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        losses = []
        #self.update_parameters(self.persionalized_model_bar)
        for user, item, label in self.trainloader:
            user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)
            prediction = self.model(user, item)

            # reg_loss = self.reg_loss(self.model, self.p_local_model)
            loss = self.loss(prediction, label)
            losses.append(loss.item())

        #self.update_parameters(self.local_model)

        return sum(losses) / len(losses)
#############################################################


##################### dateset next batch #####################
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (user, item, label) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (user, item, label) = next(self.iter_trainloader)
        return (user, item, label)

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (user, item, label) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (user, item, label) = next(self.iter_testloader)
        return (user.to(self.device), item.to(self.device), label.to(self.device))
#############################################################


###################### user evaluate ######################
    # pFedMe
    def user_persionalized_evaluate(self, epoch, train_loss):
        HR, NDCG = self.test_persionalized_model()
        # train_loss = self.train_error_and_loss_persionalized_model()

        print("user: {}   loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(self.id, train_loss, HR, NDCG))

        user_logging(epoch, self.id, train_loss, HR, NDCG, self.running_time, self.hyper_param)

    def user_evaluate(self, epoch, train_loss):
        HR, NDCG = self.test()
        # train_loss = self.train_error_and_loss()

        print("user: {}   loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(self.id, train_loss, HR, NDCG))

        user_logging(epoch, self.id, train_loss, HR, NDCG, self.running_time, self.hyper_param)
#############################################################




















