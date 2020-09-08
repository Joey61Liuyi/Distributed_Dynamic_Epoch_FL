#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt        #matplotlib inline
import csv

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
from DNC_PPO import PPO

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
        self.history_avg_price = np.zeros(5)

    def reset(self):
        self.index = 0
        self.state = 0.0001 * self.data_size + self.frequency  #TODO
        self.state_ = np.zeros(configs.user_num)
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
        self.test_acc_before = 0
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

        # TODO  DRL Action

        action = 5 * action
        action = action.astype(int)

        if ((action==[0,0,0,0,0]).all()):
            action = [0,0,0,1,0]
        print("Action", action)

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
        #     print(f'Training Loss : {np.mean(np.array(self.train_loss))}')train_accuracy
        #     print('Train Accuracy: {:.2f}% \n'.format(100 * self.[-1]))


        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        delta_acc = test_acc - self.test_acc_before # acc increment for reward
        self.test_acc_before = test_acc

        print(f' \nAvg Training Stats after {self.index + 1} global rounds:')
        print(f'Test Loss: {test_loss}')
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))


        self.index += 1


        # TODO     Env for Computing Time & State Transition & Reward Design

        time_cmp = (action * self.D * self.C) / self.frequency
        print("Computing Time:", time_cmp)

        time_global = np.max(time_cmp)
        print("Global Time:", time_global)

        payment = np.dot(action, self.state)
        print("Payment:", payment)

        print("Accuracy:", test_acc, "Accuracy increment:", delta_acc)

        reward = (self.lamda*delta_acc - payment - time_global) / 10    #TODO reward percentage need to be change
        print("Scaling Reward:", reward)
        print("###################################################################")


        # # todo state transition here
        #
        # for i in range(self.state.size):
        #     if action[i] == 0:
        #         # user will decrease its price to join next round if not join the training in this round
        #         self.state[i] = 0.8 * self.state[i]
        #     else:
        #         if self.state[i] * action[i] >= self.history_avg_price[i]:
        #             # if user's current revenue >= history revenue, it wants to increase price to get more
        #             self.state_[i] = 1.05 * self.state[i]
        #             self.history_avg_price[i] = (self.history_avg_price[i]+self.state[i] * action[i]) / 2
        #         else:
        #             # if user's current revenue < history revenue, it wants to increase price to get more
        #             self.state[i] = 0.95 * self.state[i]
        #             self.history_avg_price[i] = (self.history_avg_price[i] + self.state[i] * action[i]) / 2
        #
        # self.state = self.state_

        return reward, self.state, test_acc, payment, time_global

# TODO  The above is Environment



# TODO  The below is main DRL training progress

if __name__ == '__main__':

    configs = Configs()
    env = Env(configs)
    ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS, configs.HAVE_TRAIN, 0)

    csvFile1 = open("recording2-Dynamic-local-epoch_" + "Client_" + str(configs.user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)

    accuracies = []
    payments = []
    round_times = []

    rewards = []
    actions = []
    closses = []
    alosses = []



    for EP in range(configs.EP_MAX):
        cur_state = env.reset()
        observation = cur_state
        recording = []

        #  learning rate change for trade-off between exploit and explore
        if EP % 10 == 0:
            dec =  configs.dec * 0.95
            A_LR = configs.A_LR * 0.85
            C_LR = configs.C_LR * 0.85

        buffer_s = []
        buffer_a = []
        buffer_r = []
        sum_accuracy = 0
        sum_payment = 0
        sum_round_time =0
        sum_reward = 0
        sum_action = 0
        sum_closs = 0
        sum_aloss = 0

        for t in range(configs.rounds):
            # local_ep_list = input('please input the local epoch list:')
            # local_ep_list = local_ep_list.split(',')
            # local_ep_list = [int(i) for i in local_ep_list]
            # action = local_ep_list
            action = ppo.choose_action(observation, configs.dec)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(action)
            # while action == np.array([0,0,0,0,0]):
            #     action = ppo.choose_action(observation, configs.dec)
            reward, next_state, accuracy, pay, round_time = env.step(action)

            sum_accuracy += accuracy
            sum_payment += pay
            sum_round_time += round_time
            sum_reward += reward
            sum_action += action
            buffer_a.append(action.copy())
            buffer_s.append(cur_state.reshape(-1, configs.S_DIM).copy())
            buffer_r.append(reward)

            cur_state = next_state

            #  ppo.update()
            if (t+1) % configs.BATCH == 0:
                discounted_r = np.zeros(len(buffer_r), 'float32')
                v_s = ppo.get_v(next_state.reshape(-1, configs.S_DIM))
                running_add = v_s

                for rd in reversed(range(len(buffer_r))):
                    running_add = running_add * configs.GAMMA + buffer_r[rd]
                    discounted_r[rd] = running_add

                discounted_r = discounted_r[np.newaxis, :]
                discounted_r = np.transpose(discounted_r)
                if configs.HAVE_TRAIN == False:
                    closs, aloss = ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, configs.dec, configs.A_LR, configs.C_LR, EP)
                    sum_closs += closs
                    sum_aloss += aloss

        if (EP+1) % 1 == 0:
            print("------------------------------------------------------------------------")
            print('instant ep:', EP)

            rewards.append(sum_reward * 10)
            # actions.append(sum_action / configs.rounds)
            closses.append(sum_closs / configs.rounds)
            alosses.append(sum_aloss / configs.rounds)
            accuracies.append(sum_accuracy)
            payments.append(sum_payment)
            round_times.append(sum_round_time)

            recording.append(sum_reward * 10)
            # recording.append(np.floor(5*(sum_action / configs.rounds)))
            recording.append(sum_closs / configs.rounds)
            recording.append(sum_aloss / configs.rounds)
            recording.append(sum_accuracy)
            recording.append(sum_payment)
            recording.append(sum_round_time)
            writer1.writerow(recording)

            print("average reward:", sum_reward * 10)
            # print("average action:", sum_action / configs.rounds)
            print("average closs:", sum_closs / configs.rounds)
            print("average aloss:", sum_aloss / configs.rounds)
            print("average accuracy:", sum_accuracy)
            print("average payment:", sum_payment)
            print("average round time:", sum_round_time)

    plt.plot(rewards)
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    # plt.plot(actions)
    # plt.ylabel("action")
    # plt.xlabel("Episodes")
    # # plt.savefig("actions.png", dpi=200)
    # plt.show()

    plt.plot(alosses)
    plt.ylabel("aloss")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(closses)
    plt.ylabel("closs")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(payments)
    plt.ylabel("payment")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(round_times)
    plt.ylabel("round time")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(accuracies)
    plt.ylabel("accuracy")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    # writer1.writerow(rewards)
    # writer1.writerow(actions)
    # writer1.writerow(alosses)
    # writer1.writerow(closses)
    # writer1.writerow(accuracies)
    # writer1.writerow(payments)
    # writer1.writerow(round_times)
    csvFile1.close()

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





