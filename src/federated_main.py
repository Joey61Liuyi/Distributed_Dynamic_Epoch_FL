#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from RL_brain import PPO
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
import pandas as pd
import random
import threading

from configs import Configs

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Env(object):

    def __init__(self, configs):
        self.configs = configs
        self.data_size = configs.data_size
        self.frequency = configs.frequency
        self.C = configs.C
        self.lamda = configs.lamda
        self.seed = 0
        self.D = configs.D

    def reset(self):
        self.index = 0
        self.state = 0.0001 * self.data_size + self.frequency  #TODO
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)

        start_time = time.time()
        self.acc_list = []
        self.loss_list = []
        # define paths
        path_project = os.path.abspath('..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        exp_details(self.args)

        if configs.gpu:
            # torch.cuda.set_device(self.args.gpu)
            # device = 'cuda' if args.gpu else 'cpu'
            #

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)


        # BUILD MODEL
        if self.args.model == 'cnn':
            # Convolutional neural netork
            if self.args.dataset == 'mnist':
                self.global_model = CNNMnist(args=self.args)
            elif self.args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=self.args)
            elif self.args.dataset == 'cifar':
                self.global_model = CNNCifar(args=self.args)

        elif self.args.model == 'mlp':
            # Multi-layer preceptron
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                self.global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=self.args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.

        print(get_parameter_number(self.global_model))
        print('---------------------------------------------------------------------------------------')

        self.global_model.to(device)
        self.global_model.train()
        print(self.global_model)

        # copy weights
        global_weights = self.global_model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 1
        val_loss_pre, counter = 0, 0

        return self.state

    def individual_train(self, idx):
        local_ep = self.local_ep_list[idx]

        if local_ep != 0:
            local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                      idxs=self.user_groups[idx], logger=self.logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=local_ep)
            self.local_weights.append(copy.deepcopy(w))
            self.local_losses.append(copy.deepcopy(loss))



    def step(self, action):

        self.local_weights, self.local_losses = [], []
        print(f'\n | Global Training Round : {self.index + 1} |\n')

        pass
        self.global_model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

        print(idxs_users)
        # local_ep_list = input('please input the local epoch   list:')
        # local_ep_list = local_ep_list.split(',')
        # local_ep_list = [int(i) for i in local_ep_list]

        # todo  DRL Action
        self.local_ep_list = action

        thread_list = []

        for idx in idxs_users:
            thread = threading.Thread(target=self.individual_train, args=(idx,))
            thread_list.append(thread)
            thread.start()

        for i in thread_list:
            i.join()

        # update global weights
        global_weights = average_weights(self.local_weights)

        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(global_weights)

        # update global weights
        self.global_model.load_state_dict(global_weights)

        loss_avg = sum(self.local_losses) / len(self.local_losses)
        self.train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        self.global_model.eval()
        for c in range(self.args.num_users):
            local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                      idxs=self.user_groups[idx], logger=self.logger)
            acc, loss = local_model.inference(model=self.global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        self.train_accuracy.append(sum(list_acc) / len(list_acc))

        self.loss_list.append(np.mean(np.array(self.train_loss)))
        self.acc_list.append(np.mean(np.array(self.train_accuracy)))

        info = pd.DataFrame([self.acc_list, self.loss_list])
        info = pd.DataFrame(info.values.T, columns=['acc', 'loss'])
        info.to_csv(
            str(self.args.num_users) + 'user_' + self.args.dataset + '_' + str(self.args.lr) + '.csv')

        # print global training loss after every 'i' rounds

        # if (self.index + 1) % self.print_every == 0:
        #     print(f' \nAvg Training Stats after {self.index+ 1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(self.train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100 * self.train_accuracy[-1]))


        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)

        print(f' \nAvg Training Stats after {self.index + 1} global rounds:')
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        self.index += 1

        time_cmp = (action * self.D * self.C) / self.frequency
        time_globle = np.max(time_cmp)
        payment = np.dot(action, self.state)
        reward = self.lamda*test_acc - payment - time_globle

        state_ = self.state + payment  # todo state transition here for later
        self.state = state_

        return reward, self.state



if __name__ == '__main__':

    configs = Configs()
    env = Env(configs)
    ppo = PPO()


    for EP in range(configs.EP_MAX):
        cur_state = env.reset()

        for t in range(configs.rounds):
            local_ep_list = input('please input the local epoch list:')
            local_ep_list = local_ep_list.split(',')
            local_ep_list = [int(i) for i in local_ep_list]
            action = local_ep_list
            # action = PPO.choose_action(cur_state)
            reward, state_ = env.step(action)

        # ppo.update()



    # TODO Inference with test data

    # Test inference after completion of training
    #
    #
    #
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)
    #
    #
    #
    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    #
    # # Saving the objects train_loss and train_accuracy:
    # file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)





