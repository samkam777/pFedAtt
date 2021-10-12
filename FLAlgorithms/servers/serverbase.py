import torch
import os
import numpy as np
import copy
from logging_results import server_logging
import torch.nn.functional as F

class Server:
    def __init__(self, device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time, hyper_param):

        self.device = device
        self.num_glob_iters = args.epochs
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = args.seg_data
        self.beta = args.beta
        self.lamda = args.lamda
        self.rs_train_loss_per, self.rs_HR_per, self.rs_NDCG_per, self.rs_train_loss, self.rs_HR, self.rs_NDCG = [], [], [], [], [], []
        self.times = times
        self.running_time = running_time
        self.hyper_param = hyper_param
        # DP
        self.if_DP = args.if_DP

############## get and set server parameters ##############
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
    
    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio#################################

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def get_att_weight(self, user):
        att_weight = torch.tensor(0.).to(self.device)
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            att_weight += torch.norm(server_param-user_param, p=2)
        return att_weight
############################################################


################# weight parameters update #################
    # FedAvg aggregate
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)


    # pFedMe
    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)            

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data      

    # attention pFed
    def attention_persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        att_w = []
        for user in self.selected_users:
            att_weight = self.get_att_weight(user)
            att_w.append(att_weight)
        att_w_ = torch.Tensor(att_w)
        print("att_w: {}\t".format(att_w_))
        min_att_w_ = torch.min(att_w_)
        max_att_w_ = torch.max(att_w_)
        # norm_att_w_ = 1 - ((att_w_ - min_att_w_) / (max_att_w_ - min_att_w_))
        norm_att_w_ = (att_w_ - min_att_w_) / (max_att_w_ - min_att_w_)
        norm_att_w_ = F.softmax(norm_att_w_, dim=0)   # 行和
        print("att_w after softmax: {}\t".format(norm_att_w_))

        for i, user in enumerate(self.selected_users):
            self.add_parameters(user, norm_att_w_[i])  

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data   
        

    def select_users(self, round, num_users):
        return self.users
############################################################


###################### server evaluate ######################
    # FedAvg
    def test(self):
        total_HR = []
        total_NDCG = []
        for c in self.users:
            HR, NDCG = c.test()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # print("testing global client {}  HR: {:.4f}  NDCG: {:.4f}\t".format(c.id, HR, NDCG))
        ids = [c.id for c in self.users]

        return ids, total_HR, total_NDCG

    # pFedMe
    def test_persionalized_model(self):
        total_HR = []
        total_NDCG = []
        for c in self.users:
            HR, NDCG = c.test_persionalized_model()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # print("testing persionalized model client {}  HR: {:.4f}  NDCG: {:.4f}\t".format(c.id, HR, NDCG))
        ids = [c.id for c in self.users]

        return ids, total_HR, total_NDCG

    # pFedMe
    def train_error_and_loss_persionalized_model(self):
        losses = []
        for c in self.users:
            loss = c.train_error_and_loss_persionalized_model() 
            losses.append(loss)

        ids = [c.id for c in self.users]
        return ids, losses

    # FedAvg
    def train_error_and_loss(self):
        losses = []
        for c in self.users:
            loss = c.train_error_and_loss() 
            losses.append(loss)

        ids = [c.id for c in self.users]
        return ids, losses


    # FedAvg
    def evaluate(self, epoch, losses):
        stats = self.test()
        # stats_train = self.train_error_and_loss()

        HR = sum(stats[1]) / len(stats[1])
        NDCG = sum(stats[2]) / len(stats[2])
        train_loss = sum(losses) / len(losses)

        print("Average server   loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(train_loss, HR, NDCG))

        server_logging(epoch, train_loss, HR, NDCG, self.running_time, self.hyper_param)

    # pFedMe
    def evaluate_personalized_model(self, epoch, losses):
        stats = self.test_persionalized_model()  
        # stats_train = self.train_error_and_loss_persionalized_model()

        HR = sum(stats[1]) / len(stats[1])
        NDCG = sum(stats[2]) / len(stats[2])
        train_loss = sum(losses) / len(losses)

        print("Personal server   loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(train_loss, HR, NDCG))

        server_logging(epoch, train_loss, HR, NDCG, self.running_time, self.hyper_param)
#############################################################



















