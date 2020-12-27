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
from itertools import product

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
        self.history_avg_price = np.zeros(self.configs.user_num)

    def reset(self):
        self.index = 0
        self.data_value = 0.001 * self.data_size
        self.unit_E = self.configs.frequency * self.configs.frequency * self.configs.C * self.configs.D * self.configs.alpha  #TODO
        self.bid = self.data_value + self.unit_E
        self.bid_ = np.zeros(self.configs.user_num)
        self.action_history = []
        # self.bid_min = 0.7 * self.bid

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

        if self.configs.gpu:
            # torch.cuda.set_device(self.args.gpu)
            # device = 'cuda' if args.gpu else 'cpu'

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)
        if self.configs.remove_client_index != None:
            self.user_groups.pop(self.configs.remove_client_index)

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
        self.loss_before = 300
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 1
        val_loss_pre, counter = 0, 0

        return self.bid


    # TODO   for multi- thread
    # def individual_train(self, idx):
    #     local_ep = self.local_ep_list[idx]
    #
    #     if local_ep != 0:
    #         local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
    #                                   idxs=self.user_groups[idx], logger=self.logger)
    #         w, loss = local_model.update_weights(
    #             model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=local_ep)
    #         self.local_weights.append(copy.deepcopy(w))
    #         self.local_losses.append(copy.deepcopy(loss))

    def fake_step(self):

        weights_rounds, local_losses = [], []
        print(f'\n | Global Training Round : {self.index + 1} |\n')

        global_model_tep = copy.deepcopy(self.global_model)

        global_model_tep.train()

        idxs_users = list(self.user_groups.keys())

        # Local Training


        possible_epochs = list(range(1,self.configs.myopia_max_epoch+1))
        for epoch in possible_epochs:
            weights_users = []
            for idx in idxs_users:

                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx], logger=self.logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=epoch)
                weights_users.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            weights_rounds.append(copy.deepcopy(weights_users))

        possible_epochs = list(range(self.configs.myopia_max_epoch+1))
        loop_val = []
        for i in range(self.configs.user_num):
            loop_val.append(possible_epochs)

        result_book = pd.DataFrame([], columns=["action", "reward"], index=None)

        for i in product(*loop_val):
            if random.uniform(0, 1) > self.configs.myopia_frac:
                continue
            weights_tep = []
            action = list(i)
            for one in action:
                if one:
                    weights_tep.append(weights_rounds[one-1][action.index(one)])
            if weights_tep != []:
                global_weights = average_weights(weights_tep)
                global_model_tep = copy.deepcopy(self.global_model)
                global_model_tep.load_state_dict(global_weights)
                global_model_tep.eval()
                test_acc, test_loss = test_inference(self.args, global_model_tep, self.test_dataset)

                delta_acc = test_acc - self.acc_before
                delta_loss = self.loss_before - test_loss
                action = np.array(action)
                time_cmp = (action * self.D * self.C) / self.frequency
                time_global = np.max(time_cmp)

                data_value_sum = np.dot(action, self.data_value)
                E = np.dot(action, self.unit_E)
                cost = data_value_sum + E

                if self.configs.performance == 'acc':
                    delta_performance = delta_acc
                else:
                    delta_performance = delta_loss

                reward = (self.lamda * delta_performance - cost)/10 #TODO test for the existance of data importance

                print(action, reward)
                result_book = result_book.append([{'action': action, 'reward': reward}])

        result_book.to_csv('Result_book_of_round_'+str(self.index)+'.csv', index=None)

        return result_book.sort_values('reward').iloc[-1]['action'], result_book.sort_values('reward').iloc[-1]['reward']

    def step(self, action):


        self.local_weights, self.local_losses = [], []
        print(f'\n | Global Training Round : {self.index + 1} |\n')

        pass
        self.global_model.train()
        idxs_users = np.array(list(self.user_groups.keys()))
        print("User index:",idxs_users)

        # TODO  DRL Action

        action = 5 * action
        action = action.astype(int)

        #TODO FedAvg here
        # tep = 3
        # action = np.array([tep, tep, tep, tep, tep])

        self.action_history = list(self.action_history)
        self.action_history.append(action)
        self.action_history = np.array(self.action_history)

        print("Action", action)
        print(type(action))
        self.local_ep_list = action


        #TODO single thread
        for idx in idxs_users:

            local_ep = self.local_ep_list[list(idxs_users).index(idx)]

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
        # From now on, set the model to evaluation
        self.global_model.eval()

        for idx in idxs_users:
            local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                      idxs=self.user_groups[idx], logger=self.logger)
            acc, loss = local_model.inference(model=self.global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        self.train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds

        # delta_acc = np.mean(np.array(self.train_accuracy)) - self.acc_before
        # self.acc_before = np.mean(np.array(self.train_accuracy))


        if (self.index + 1) % self.print_every == 0:
            print(f' \nAvg Training Stats after {self.index+ 1} global rounds:')
            # print(f'Training Loss : {np.mean(np.array(self.train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100 * np.mean(np.array(self.train_accuracy))))
            print(f'Training Loss : {self.train_loss[-1]}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * self.train_accuracy[-1]))


        # TODO    test accuracy

        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        self.test_accuracy.append(test_acc)
        self.test_loss.append(test_loss)

        delta_acc = self.test_accuracy[-1] - self.acc_before
        self.acc_before = self.test_accuracy[-1]

        delta_loss = self.loss_before - self.test_loss[-1]
        self.loss_before = self.test_loss[-1]
        print("Loss:", self.test_loss[-1], "Loss increment:", delta_loss)
        print("Accuracy:", self.test_accuracy[-1], "Accuracy increment:", delta_acc)

        # test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        # delta_acc = test_acc - self.test_acc_before # acc increment for reward
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

        # E = configs.frequency * configs.frequency * configs.C * configs.D * configs.alpha
        # E = E * action
        # E = np.sum(E)

        data_value_sum = np.dot(action, self.data_value)
        print("Sum Data Value:", data_value_sum)

        E = np.dot(action, self.unit_E)
        print("Energy:", E)

        cost = data_value_sum + E
        print("cost:", cost)


        if self.configs.performance == 'acc':
            delta_performance = delta_acc
        else:
            delta_performance = delta_loss
        # reward = (self.lamda * delta_acc - payment - time_global) / 10   #TODO reward percentage need to be change
        reward = (self.lamda * delta_performance - cost)/10 #TODO test for the existance of data importance
        print("Scaling Reward:", reward)
        print("------------------------------------------------------------------------")

        # todo state transition here

        print("########################################################################")
        print("Action History:", self.action_history)
        history_cut = self.action_history[-3:]
        print("History Cut:", history_cut)
        history_avg = np.mean(history_cut, axis=0)
        print("history_avg:", history_avg)

        print("Data Value before:", self.data_value)
        sign_add = action > history_avg
        print("Sign Add:", sign_add)
        sign_reduce = action < history_avg
        print("Sign Reduce:", sign_reduce)
        self.data_value = self.data_value * sign_add * 0.1 - self.data_value * sign_reduce * 0.1 + self.data_value
        print("Data Value after:", self.data_value)

        self.bid_ = self.data_value + self.unit_E
        print("Bid:", self.bid)
        print("Next Bid:", self.bid_)

        # for i in range(self.bid.size):
        #
        #     if action[i] > history_avg[i]:
        #         self.bid_[i] = 1.1 * self.bid[i]
        #     elif action[i] < history_avg[i]:
        #         self.bid_[i] = 0.9 * self.bid[i]
        #     else:
        #         self.bid_[i] = self.bid[i]

        self.bid = self.bid_

        return reward, self.bid, delta_performance, cost, time_global, action, E

# TODO  The above is Environment


def fed_avg():
    configs = Configs()
    env = Env(configs)
    env.reset()
    data = pd.DataFrame([], columns=['action', 'reward', 'delta_accuracy', 'round_time', 'energy'])


    for one in range(configs.rounds):
        action = []
        for i in range(configs.user_num):
            action.append(1)
        action = np.array(action) / 5
        reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)
        data = data.append([{'action': action, 'reward': reward, 'delta_accuracy': delta_accuracy,
                             'round_time': round_time, 'energy': energy}])
    data.to_csv('fed_avg1.csv', index=None)

def Greedy_myopia():
    configs = Configs()
    env = Env(configs)
    env.reset()
    data = pd.DataFrame([], columns=['action','reward', 'delta_accuracy', 'round_time', 'energy'])

    for one in range(configs.rounds):
        action, reward = env.fake_step()
        action = np.array(action)/5
        reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)
        data = data.append([{'action': action, 'reward': reward, 'delta_accuracy': delta_accuracy, 'round_time': round_time, 'energy': energy}])
    data.to_csv('Greedy_myopia.csv', index=None)

def DRL_inference(agent_info):
    configs = Configs()
    env = Env(configs)

    ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS, True, agent_info)
    recording = pd.DataFrame([], columns=['state history', 'action history', 'reward history', 'acc increase hisotry', 'time hisotry', 'energy history', 'social welfare', 'accuracy', 'time', 'energy'])


    for EP in range(50):
        cur_bid = env.reset()
        cur_state = np.append(cur_bid, env.index)

        state_list = []
        action_list = []
        reward_list = []
        performance_increase_list = []
        time_list = []
        energy_list = []
        for t in range(configs.rounds):
            print("Current State:", cur_state)
            action = ppo.choose_action(cur_state, configs.dec)
            while (np.floor(5 * action) == np.zeros(configs.user_num, )).all():
                action = ppo.choose_action(cur_state, configs.dec)
            action[4] = 0
            print(action)
            reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

            cur_bid = next_bid
            next_state = np.append(next_bid, env.index)
            cur_state = next_state

            state_list.append(cur_state)
            action_list.append(int_action)
            reward_list.append(reward)
            performance_increase_list.append(delta_accuracy)
            time_list.append(round_time)
            energy_list.append(energy)

        recording = recording.append([{'state history': state_list, 'action history': action_list, 'reward history':reward_list, 'acc increase hisotry': performance_increase_list, 'time hisotry': time_list, 'energy history': energy_list, 'social welfare': np.sum(reward_list), 'accuracy': np.sum(performance_increase_list), 'time': np.sum(time_list), 'energy': np.sum(energy_list)}])
        recording.to_csv(agent_info+'_Inference result.csv')

def DRL_train():

    configs = Configs()
    env = Env(configs)
    agent_info = str(configs.remove_client_index)+configs.data+'_'+configs.performance + time.strftime("%Y-%m-%d", time.localtime())
    ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS, configs.HAVE_TRAIN, agent_info)
    #todo num=0 2rounds on GPU; num=1 10rounds; num=2 20rounds of TestAcc; num=3 10Rounds test for data importance

    csvFile1 = open("Loss-State(Action Avg)" + "Client_" + str(configs.user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)

    accuracies = []
    costs = []
    round_times = []

    rewards = []
    closses = []
    alosses = []
    dec = configs.dec
    A_LR = configs.A_LR
    C_LR = configs.C_LR
    C_loss = pd.DataFrame(columns=['Episodes', 'C-loss'])

    for EP in range(configs.EP_MAX):
        cur_bid = env.reset()
        cur_state = np.append(cur_bid, 0)  #TODO  add index into state
        recording = []
        recording.append(cur_state)

        #  learning rate change for trade-off between exploit and explore
        if EP % 20 == 0:
            dec = dec * 0.95
            A_LR = A_LR * 0.85
            C_LR = C_LR * 0.85

        buffer_s = []
        buffer_a = []
        buffer_r = []
        sum_accuracy = 0
        sum_cost = 0
        sum_round_time = 0
        sum_reward = 0
        sum_action = 0
        sum_closs = 0
        sum_aloss = 0
        sum_energy = 0

        for t in range(configs.rounds):
            # local_ep_list = input('please input the local epoch list:')
            # local_ep_list = local_ep_list.split(',')
            # local_ep_list = [int(i) for i in local_ep_list]
            # action = local_ep_list
            print("Current State:", cur_state)

            action = ppo.choose_action(cur_state, configs.dec)
            while (np.floor(5*action) == np.zeros(configs.user_num,)).all():
                action = ppo.choose_action(cur_state, configs.dec)

            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(action)
            # while action == np.array([0,0,0,0,0]):
            #     action = ppo.choose_action(observation, configs.dec)
            reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

            # next_bid = cur_bid  # todo Fix biding, to be deleted after trial experiment

            sum_accuracy += delta_accuracy
            sum_cost += cost
            sum_round_time += round_time
            sum_reward += reward
            sum_action += action
            sum_energy += energy
            buffer_a.append(action.copy())
            buffer_r.append(reward)
            buffer_s.append(cur_state.reshape(-1, configs.S_DIM).copy())

            next_state = np.append(next_bid, t+1)
            recording.append(int_action)
            recording.append(reward)
            recording.append(next_state)

            print("Current State:", cur_state)
            print("Next State:", next_state)

            #  ppo.update()
            if (t+1) % configs.BATCH == 0:
                print("------------PPO UPDATED------------")
                discounted_r = np.zeros(len(buffer_r), 'float32')
                v_s = ppo.get_v(next_state.reshape(-1, configs.S_DIM))
                running_add = v_s

                for rd in reversed(range(len(buffer_r))):
                    running_add = running_add * configs.GAMMA + buffer_r[rd]
                    discounted_r[rd] = running_add

                discounted_r = discounted_r[np.newaxis, :]
                discounted_r = np.transpose(discounted_r)
                if configs.HAVE_TRAIN == False:
                    closs, aloss = ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, configs.dec, configs.A_LR, configs.C_LR, EP+1)
                    sum_closs += closs
                    sum_aloss += aloss
                    C_loss.append([{'Episodes': EP, 'C-loss': closs}])

            #TODO state transition
            cur_state = next_state
            print("################################# ROUND END #####################################")

        if (EP+1) % 1 == 0:
            print("------------------------------------------------------------------------")
            print('instant ep:', (EP+1))

            rewards.append(sum_reward * 10)
            # actions.append(sum_action / configs.rounds)
            closses.append(sum_closs / configs.rounds)
            alosses.append(sum_aloss / configs.rounds)
            accuracies.append(sum_accuracy)
            costs.append(sum_cost)
            round_times.append(sum_round_time)

            recording.append(sum_reward * 10)
            # recording.append(np.floor(5*(sum_action / configs.rounds)))
            recording.append(sum_closs / configs.rounds)
            recording.append(sum_aloss / configs.rounds)
            recording.append(sum_accuracy)
            recording.append(sum_cost)
            recording.append(sum_round_time)
            recording.append(sum_energy)
            writer1.writerow(recording)

            print("accumulated reward:", sum_reward * 10)
            # print("average action:", sum_action / configs.rounds)
            print("average closs:", sum_closs / configs.rounds)
            print("average aloss:", sum_aloss / configs.rounds)
            print("total accuracy:", sum_accuracy)
            print("total cost:", sum_cost)
            print("total round time:", sum_round_time)

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

    plt.plot(costs)
    plt.ylabel("cost")
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
    C_loss.to_csv('DRL_closs.csv', index=None)

def Hand_control():
    configs = Configs()
    env = Env(configs)
    recording = pd.DataFrame([], columns=['state history', 'action history', 'reward history', 'acc increase hisotry', 'time hisotry', 'energy history', 'social welfare', 'accuracy', 'time', 'energy'])

    cur_bid = env.reset()
    cur_state = np.append(cur_bid, env.index)

    state_list = []
    action_list = []
    reward_list = []
    performance_increase_list = []
    time_list = []
    energy_list = []
    for t in range(configs.rounds):
        print("Current State:", cur_state)
        local_ep_list = input('please input the local epoch list:')
        local_ep_list = local_ep_list.split(',')
        local_ep_list = [int(i) for i in local_ep_list]
        action = np.array(local_ep_list)/5
        print(action)
        reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

        cur_bid = next_bid
        next_state = np.append(next_bid, env.index)
        cur_state = next_state

        state_list.append(cur_state)
        action_list.append(int_action)
        reward_list.append(reward)
        performance_increase_list.append(delta_accuracy)
        time_list.append(round_time)
        energy_list.append(energy)

    recording = recording.append([{'state history': state_list, 'action history': action_list, 'reward history':reward_list, 'acc increase hisotry': performance_increase_list, 'time hisotry': time_list, 'energy history': energy_list, 'social welfare': np.sum(reward_list), 'accuracy': np.sum(performance_increase_list), 'time': np.sum(time_list), 'energy': np.sum(energy_list)}])
    recording.to_csv('Hand_control_result.csv')

def greedy():
    configs = Configs()
    env = Env(configs)
    csvFile1 = open("recording-Greedy_" + "Client_" + str(configs.user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)

    accuracies = []
    payments = []
    round_times = []
    rewards = []
    Actionset_list = []

    for EP in range(configs.EP_MAX):

        cur_bid = env.reset()
        cur_state = np.append(cur_bid, 0)

        recording = []
        recording.append(cur_state)

        sum_accuracy = 0
        sum_cost = 0
        sum_round_time = 0
        sum_reward = 0
        sum_energy = 0

        if len(Actionset_list) < 20:    # action in first 20 episode is randomly chose
            actionset = np.random.random(configs.rounds * configs.A_DIM)
            actionset = actionset.reshape(configs.rounds, configs.A_DIM)
        else:
            tep = np.random.random(1)[0]
            if tep <= 0.2:    # 20% to randomly choose action
                actionset = np.random.random(configs.rounds * configs.A_DIM)
                actionset = actionset.reshape(configs.rounds, configs.A_DIM)
            else:     # 80% to choose the Max-R action (Greedy)
                actionset = Actionset_list[0][0]
        print("ActionSet:", actionset)

        for t in range(configs.rounds):
            action = actionset[t]
            reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

            sum_accuracy += delta_accuracy
            sum_cost += cost
            sum_round_time += round_time
            sum_reward += reward
            sum_energy += energy

            next_state = np.append(next_bid, t + 1)


            recording.append(int_action)
            recording.append(reward)
            recording.append(next_state)


            next_state = np.append(next_bid, t+1)
            cur_state = next_state

        # if action-set is unchanged (80% greedy), then remove it and re-add it with its new reward in this round
        # if action-set is changed (20% random), then add it to the actionset-list
        for one in Actionset_list:
            if (one[0] == actionset).all():
                Actionset_list.remove(one)

        # add the actionset in this round and sort actionset-list by Reward in descending order
        Actionset_list.append((actionset, sum_reward))
        Actionset_list = sorted(Actionset_list, key=lambda x: x[1], reverse=True)
        print("ActionSet-List:", Actionset_list)

        # if action-set is unchanged (80% greedy),the actionset-list = 20 and no one will pop
        # if action-set is changed (20% random), pop the last actionset (sorted with Min-R)
        if len(Actionset_list) > 20:
            Actionset_list.pop()

        if (EP+1) % 1 == 0:
            print("------------------------------------------------------------------------")
            print('instant ep:', (EP+1) )

            recording.append(sum_reward * 10)
            recording.append(sum_accuracy)
            recording.append(sum_cost)
            recording.append(sum_round_time)
            recording.append(sum_energy)
            writer1.writerow(recording)
    csvFile1.close()


if __name__ == '__main__':
    # DRL_train()
    # fed_avg()
    # DRL_inference('mnist_acc2020-12-01')
    # Greedy_myopia()
    # Hand_control()
    greedy()
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
