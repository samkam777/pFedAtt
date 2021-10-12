import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

from m_opacus.dp_model_inspector import DPModelInspector
from m_opacus.utils import module_modification
from m_opacus import PrivacyEngine

class UserAVG(User):
    def __init__(self, device, args, model, train_data, test_data, train_data_samples, numeric_id, running_time, hyper_param):
        super().__init__(device, args, model, train_data, test_data, train_data_samples, numeric_id, running_time, hyper_param)
        self.total_users = args.seg_data

        self.loss = nn.BCELoss()
        self.personal_learning_rate = args.personal_learning_rate
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.personal_learning_rate, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.personal_learning_rate)
        if self.if_DP:
            self.privacy_engine = PrivacyEngine(
                self.model,
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
        self.model.train()
        self.optimizer.zero_grad()
        losses = []

        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            user, item, label = self.get_next_train_batch()

            #for user, item, label in self.trainloader:
            user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)
            prediction = self.model(user, item)
            loss = self.loss(prediction, label)
            loss.backward()


            if self.if_DP:
            # 只有最后一个epoch加噪声
                if(epoch == self.local_epochs):
                    print("add noise!")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.original_step()
                    self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.clone_model_paramenter(self.model.parameters(), self.local_model)  # clone model to local_model

            losses.append(loss.item())
        
        # calculate loss
        train_loss = sum(losses) / len(losses)

        # evaluate
        self.user_evaluate(epochs, train_loss)

        # calculate privacy budget
        if self.if_DP:
            self.privacy_engine.steps = epochs+1
            epsilons, _ = self.privacy_engine.get_privacy_spent(self.delta)
            print("training epochs {}  client {}  epsilons:{:.4f}\t".format(epochs, self.id, epsilons))
            return train_loss, epsilons
        else:
            return train_loss














