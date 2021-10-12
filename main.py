import os
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from FLAlgorithms.trainmodel.model import *
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from util import str2bool
import util
import data_utils

import copy
import math

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from opacus import PrivacyEngine



def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", 
		type=int, 
		default=42, 
		help="Seed")
	parser.add_argument("--lr", 
		type=float, 
		default=0.01, 
		help="learning rate")
	parser.add_argument("--dropout", 
		type=float,
		default=0.2,  
		help="dropout rate")
	parser.add_argument("--batch_size", 
		type=int, 
		default=512, 
		help="batch size for training")
	parser.add_argument("--epochs", 
		type=int,
		default=600,  
		help="training epoches")
	parser.add_argument("--top_k", 
		type=int, 
		default=10, 
		help="compute metrics@top_k")
	parser.add_argument("--factor_num", 
		type=int,
		default=32, 
		help="predictive factors numbers in the model")
	parser.add_argument("--layers",
		nargs='+', 
		default=[64,32,16,8],
		help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
	parser.add_argument("--num_ng", 
		type=int,
		default=4, 
		help="Number of negative samples for training set")
	parser.add_argument("--num_ng_test", 
		type=int,
		default=100, 
		help="Number of negative samples for test set")
	parser.add_argument("--seg_data", 
		type=int,
		default=50, 
		help="Number of segmentation of data")
	parser.add_argument("--times",
		type=int,
		default=1,
		help="running time")
	parser.add_argument("--local_epochs",
		type=int,
		default=1)
	parser.add_argument("--algorithm",
		type=str,
		default="pFedMe",
		choices=["pFedMe", "FedAvg"]) 
	# personal setting
	parser.add_argument("--personal_learning_rate",
		type=float,
		default=0.01,
		help="Persionalized learning rate to caculate theta aproximately using K steps")
	parser.add_argument("--lamda",
		type=float,
		default=0.0001,
		help="Regularization term")
	parser.add_argument("--K",
		type=int,
		default=1,
		help="Computation steps")
	parser.add_argument("--beta",
		type=float,
		default=1.0,
		help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
	# DP
	parser.add_argument('--delta',
		type=float,
		default=1e-4,
		help='DP DELTA')
	parser.add_argument('--max_grad_norm',
		type=float,
		default= 1.0,
		help='DP MAX_GRAD_NORM')
	parser.add_argument('--noise_multiplier',
		type=float,
		default= 1.0,
		help='DP NOISE_MULTIPLIER')
	parser.add_argument('--virtual_batch_size',
		type=int,
		default=512000, 
        help='DP VIRTUAL_BATCH_SIZE')
	parser.add_argument("--if_DP", 
		type=str2bool,
		default=False,
		help="if DP")
	# if sample data 重复抽样数据
	parser.add_argument("--ran_sample", 
		type=str2bool,
		default=False,
		help="if sample data")
	# if balance data
	parser.add_argument("--_balance", 
		type=str2bool,
		default=False,
		help="if balance data")
	# if subsample data	二次采样数据
	parser.add_argument("--_subsample",
		type=str2bool, 
		default=False,
		help="if subsample data")
	# get time
	parser.add_argument("--_running_time", 
        type=str, 
        default="2021-00-00-00-00-00", 
        help="running time")
	parser.add_argument("--loss_reduction", 
        type=str, 
        default="mean", 
		choices=["mean", "sum"])
	

	# set device and parameters
	args = parser.parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# seed for Reproducibility
	util.seed_everything(args.seed)

	# hyper-parameter
	hyper_param = "_balance_" + str(args._balance) + "_user_" + str(args.seg_data) + "_DP_" + str(args.if_DP) + "_algorithm_" + str(args.algorithm) + "_GlobalLr_" + str(args.lr) + "_PersonalLr_" + str(args.personal_learning_rate) + "_lamda_" + str(args.lamda) + "_loss_reduction_" + str(args.loss_reduction) + "_beta_" + str(args.beta) + "_"
	print("hyper_param: {}\t".format(hyper_param))
	

	for i in range(args.times):
		print("---------------Running time:------------",i)
		# running time
		if args._running_time != "2021-00-00-00-00-00":
			running_time = args._running_time			# get time from *.sh file 
		else:
			running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())		# get the present time
		print("running time: {}".format(running_time))

		# load client train data and test data
		ml_1m = pd.read_csv(
			r'./data/ml-1m/ratings.dat', 
			sep="::", 
			names = ['user_id', 'item_id', 'rating', 'timestamp'], 
			engine='python')
		# set the num_users, items
		num_users = ml_1m['user_id'].nunique()
		num_items = ml_1m['item_id'].nunique()
		# print("num_items: {} ".format(num_items))	# 3706
		# print("num_users: {} ".format(num_users))	# 6040

		# construct the train and test datasets
		data = data_utils.NCF_Data(args, ml_1m)
		if args._balance:
			print("balance data!!")
			train_loader, train_data_samples, total_train_data_sample = data.get_train_instance()
			test_loader = data.get_test_instance()
		else:
			print("unbalance data!!")
			train_loader, train_data_samples, total_train_data_sample = data.get_train_instance_unbalance()
			test_loader = data.get_test_instance_unbalance()

		# model
		model = NeuMF(args, num_users, num_items)
		model.to(device)

		if(args.algorithm == "pFedMe"):
			server = pFedMe(device, args, train_loader, test_loader, model, i, train_data_samples, total_train_data_sample, running_time, hyper_param)

		if(args.algorithm == "FedAvg"):
			server = FedAvg(device, args, train_loader, test_loader, model, i, train_data_samples, total_train_data_sample, running_time, hyper_param)

		server.train()
		server.test()


if __name__ == '__main__':
	main()





