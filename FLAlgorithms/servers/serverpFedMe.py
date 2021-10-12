import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
import numpy as np

from logging_results import eps_logging

class pFedMe(Server):
    def __init__(self, device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time, hyper_param):
        super().__init__(device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time, hyper_param)

        #self.k = args.K
        self.personal_learning_rate = args.personal_learning_rate
        for i in range(args.seg_data):
            # load data
            train_data = train_loader[i]
            test_data = test_loader[i]
            user = UserpFedMe(device, args, model, train_data, test_data, train_data_samples[i], i, running_time, hyper_param)
            self.users.append(user)
        self.total_train_samples = total_train_data_sample

    def train(self):
        
        
        for glob_iter in range(self.num_glob_iters):
            print("--------------------------global iter: ",glob_iter, " --------------------------")

            epsilons_list = []
            losses = []
            # do update for all users not only selected users
            for user in self.users:
                if self.if_DP:
                    train_loss, epsilons = user.train(glob_iter) # user.train_samples
                    epsilons_list.append(epsilons)
                    losses.append(train_loss)
                else:
                    train_loss = user.train(glob_iter) # user.train_samples
                    losses.append(train_loss)

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate personalized model on user for each interation
            print("")
            print("Evaluate personalized model")
            self.evaluate_personalized_model(glob_iter, losses)         # 验证的是更新模型前的
            
            # if glob_iter == 25:
            #     self.persionalized_aggregate_parameters()
            # # attention pFed
            # elif glob_iter > 25:
            #     self.attention_persionalized_aggregate_parameters()
            self.attention_persionalized_aggregate_parameters()

            self.send_parameters()

            if self.if_DP:
                eps = sum(epsilons_list) / len(epsilons_list)
                eps_logging(glob_iter, eps, self.running_time, self.hyper_param)






