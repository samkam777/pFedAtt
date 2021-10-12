import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
import numpy as np

from logging_results import eps_logging

class FedAvg(Server):
    def __init__(self, device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time, hyper_param):
        super().__init__(device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time, hyper_param)

        for i in range(args.seg_data):
            train_data = train_loader[i]
            test_data = test_loader[i]
            user = UserAVG(device, args, model, train_data, test_data, train_data_samples[i], i, running_time, hyper_param)
            self.users.append(user)
        self.total_train_samples = total_train_data_sample
    
    def train(self):

        for glob_iter in range(self.num_glob_iters):
            print("--------------------------global iter: ",glob_iter, " --------------------------")

            self.selected_users = self.select_users(glob_iter,self.num_users)
            epsilons_list = []
            losses = []
            for user in self.selected_users:
                if self.if_DP:
                    train_loss, epsilons = user.train(glob_iter) #* user.train_samples
                    epsilons_list.append(epsilons)
                    losses.append(train_loss)
                else:
                    train_loss = user.train(glob_iter) #* user.train_samples
                    losses.append(train_loss)

            print("")
            print("Evaluate average model")
            self.evaluate(glob_iter, losses)

            self.aggregate_parameters()

            self.send_parameters()

            if self.if_DP:
                eps = sum(epsilons_list) / len(epsilons_list)
                eps_logging(glob_iter, eps, self.running_time, self.hyper_param)












