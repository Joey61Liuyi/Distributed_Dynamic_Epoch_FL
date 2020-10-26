
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
        self.bid = 0.0001 * self.data_size + self.frequency  #TODO
        self.bid_ = np.zeros(configs.user_num)
        self.bid_min = 0.7 * self.bid

        # todo annotate these random seed if run greedy, save them when run DRL
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
        self.test_loss, self.test_accuracy = [], []
        self.acc_before = 0
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 1
        val_loss_pre, counter = 0, 0

        return self.bid

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

        print("User index:",idxs_users)

        # TODO  DRL Action

        action = 5 * action
        action = action.astype(int)
        print("Current Action:", action)

        self.local_ep_list = action


        #TODO single thread
        for idx in idxs_users:

            local_ep = self.local_ep_list[idx]

            if local_ep != 0:
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx], logger=self.logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=local_ep)
                self.local_weights.append(copy.deepcopy(w))
                self.local_losses.append(copy.deepcopy(loss))


        # # TODO multi-thread
        # thread_list = []
        # for idx in idxs_users:
        #     thread = threading.Thread(target=self.individual_train, args=(idx,))
        #     thread_list.append(thread)
        #     thread.start()
        #
        # for i in thread_list:
        #     i.join()


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
        self.acc_list.append(np.mean(np.array(self.train_accuracy)))   # todo np.mean(np.array(self.train_accuracy))    self.train_accuracy[-1]

        info = pd.DataFrame([self.acc_list, self.loss_list])
        info = pd.DataFrame(info.values.T, columns=['acc', 'loss'])
        info.to_csv(str(self.args.num_users) + 'user_' + self.args.dataset + '_' + str(self.args.lr) + '.csv')

        # print global training loss after every 'i' rounds

        delta_acc = np.mean(np.array(self.train_accuracy)) - self.acc_before
        # todo  np.mean(np.array(self.train_accuracy))   self.train_accuracy[-1]
        self.acc_before = np.mean(np.array(self.train_accuracy))

        # todo  np.mean(np.array(self.train_accuracy))   self.train_accuracy[-1]


        if (self.index + 1) % self.print_every == 0:
            print(f' \nAvg Training Stats after {self.index+ 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(self.train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * np.mean(np.array(self.train_accuracy))))
            # todo  np.mean(np.array(self.train_accuracy))      self.train_accuracy[-1]



        # TODO    test accuracy
        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        self.test_accuracy.append(test_acc)
        self.test_loss.append(test_loss)




        # delta_acc = test_acc - self.test_acc_before  # acc increment for reward
        # self.test_acc_before = test_acc
        #
        # print(f' \nAvg Training Stats after {self.index + 1} global rounds:')
        # print(f'Test Loss: {test_loss}')
        # print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        self.index += 1


        # TODO     Env for Computing Time & State Transition & Reward Design

        time_cmp = (action * self.D * self.C) / self.frequency
        print("Computing Time:", time_cmp)

        time_global = np.max(time_cmp)
        print("Global Time:", time_global)

        payment = np.dot(action, self.bid)
        print("Payment:", payment)

        print("Accuracy:", self.train_accuracy[-1], "Accuracy increment:", delta_acc)
        # todo  np.mean(np.array(self.train_accuracy))      self.train_accuracy[-1]

        reward = (self.lamda * delta_acc - payment - time_global) / 10    #TODO reward percentage need to be change
        print("Scaling Reward:", reward)
        print("------------------------------------------------------------------------")

        # todo state transition here

        for i in range(self.bid.size):
            if action[i] == 0:
                # user will decrease its price to join next round if not join the training in this round
                self.bid_[i] = 0.8 * self.bid[i]
                if self.bid_[i] < self.bid_min[i]:
                    self.bid_[i] = self.bid_min[i]
            else:
                if self.bid[i] * action[i] >= self.history_avg_price[i]:
                    # if user's current revenue >= history revenue, it wants to increase price to get more
                    self.bid_[i] = 1.05 * self.bid[i]
                    if self.bid_[i] < self.bid_min[i]:
                        self.bid_[i] = self.bid_min[i]
                    self.history_avg_price[i] = (self.history_avg_price[i]+self.bid[i] * action[i]) / 2
                else:
                    # if user's current revenue < history revenue, it wants to increase price to get more
                    self.bid_[i] = 0.95 * self.bid[i]
                    if self.bid_[i] < self.bid_min[i]:
                        self.bid_[i] = self.bid_min[i]
                    self.history_avg_price[i] = (self.history_avg_price[i] + self.bid[i] * action[i]) / 2

        self.bid = self.bid_

        return reward, self.bid, delta_acc, payment, time_global, action, self.train_accuracy[-1], self.train_loss[-1], self.test_accuracy[-1], self.test_loss[-1]
# TODO  The above is Environment



# TODO  The below is main DRL training progress
# todo check the random seed in Env reset !!!!!!!!!!!!!!!!!! line 53-57
if __name__ == '__main__':

    configs = Configs()
    env = Env(configs)
    ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS, True, 1)

    csvFile1 = open("recording-PPO-inference-" + "Client_" + str(configs.user_num) + "new.csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)

    accuracylist = []
    paymentlist = []
    round_timelist = []

    rewardlist = []
    actionlist = []
    statelist = []


    cur_bid = env.reset()
    cur_state = np.append(cur_bid, 0)

    sum_accuracy = 0
    sum_payment = 0
    sum_round_time = 0
    sum_reward = 0

    for rounds in range(configs.infer_round):
        recording = []
        recording.append(cur_state)
        print("Current State:", cur_state)

        # action = ppo.choose_action(cur_state, configs.dec)
        # while (np.floor(5 * action) == [0., 0., 0., 0., 0.]).all():
        #     action = ppo.choose_action(cur_state, configs.dec)

        action =np.array([0.3, 0.3, 0.3, 0.3, 0.3])


        reward, next_bid, delta_accuracy, pay, round_time, int_action, train_acc, train_loss, test_acc, test_loss = env.step(action)  #todo reward here is scaled, need to modify

        sum_accuracy += delta_accuracy
        sum_payment += pay
        sum_round_time += round_time
        sum_reward += reward

        next_state = np.append(next_bid, rounds+1)

        recording.append(int_action)
        recording.append(next_state)

        cur_state =next_state

        rewardlist.append(sum_reward*10)
        statelist.append(next_state)
        actionlist.append(action)
        accuracylist.append(sum_accuracy)
        paymentlist.append(sum_payment)
        round_timelist.append(sum_round_time)

        recording.append(sum_reward * 10)
        recording.append(sum_accuracy)
        recording.append(sum_payment)
        recording.append(sum_round_time)
        recording.append(train_acc)
        recording.append(train_loss)
        recording.append(test_acc)
        recording.append(test_loss)

        writer1.writerow(recording)

    print("accumulated reward:", sum_reward * 10)
    # print("average action:", sum_action / configs.rounds)
    print("total accuracy:", sum_accuracy)
    print("total payment:", sum_payment)
    print("total round time:", sum_round_time)

    csvFile1.close()


#
#     # TODO Inference with test data
#
#     # Test inference after completion of training
#     #
#     #
#     #
#     # test_acc, test_loss = test_inference(args, global_model, test_dataset)
#     #
#     #
#     #
#     # print(f' \n Results after {args.epochs} global rounds of training:')
#     # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
#     # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
#     #
#     # # Saving the objects train_loss and train_accuracy:
#     # file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
#     #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
#     #            args.local_ep, args.local_bs)
