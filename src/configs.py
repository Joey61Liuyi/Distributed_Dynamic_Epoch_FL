# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 18:55
# @Author  : LIU YI

import numpy as np
import pandas as pd
import math
import copy

class Configs(object):

    def __init__(self):

        ## TODO For FL training
        self.data = 'mnist'
        self.rounds = 2
        self.frac = 1
        self.user_num = 5
        self.FL_LR = 0.005
        self.model = 'cnn'
        self.iid = 0
        self.unequal = 1
        self.gpu = 0 # 0 = CPU; 1 = GPU


        # TODO for Fderated Env

        self.data_size = np.array([12000, 10000, 8000, 14000, 16000])

        if self.data == 'cifar':
            theta_num = 62006
        else:
            theta_num = 21840

        self.D = (self.data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9

        self.frequency = np.array([1.4359949, 1.52592623, 1.04966248, 1.33532239, 1.7203678])
        self.lamda = 100
        self.C = 20
        self.alpha = 0.1
        self.local_epoch_range = 10



        ## TODO For RL

        self.EP_MAX = 300
        self.S_DIM = 5  # TODO add history later
        self.A_DIM = 5
        self.BATCH = 1
        self.A_UPDATE_STEPS = 5
        self.C_UPDATE_STEPS = 5
        self.HAVE_TRAIN = False

        self.dec = 0.3
        self.A_LR = 0.00003  # todo  learning rate influence tendency
        self.C_LR = 0.00003
        self.GAMMA = 0.95
        # self.action_space = np.zeros((self.user_num, self.local_epoch_range))



#
#         self.lamda = 500
#
#         self.his_len = 5
#         self.info_num = 3
#
#
#
#         if self.data == 'cifar':
#             theta_num = 62006
#             data_size = np.array([40, 38, 32, 46, 44]) * 250 * 0.8
#
#         else:
#             theta_num = 21840
#             if self.user_num == 5:
#                 data_size = np.array([10000, 12000, 14000, 8000, 16000]) * 0.8
#             else:
#                 data_size = pd.read_csv('Multi_client_data/'+str(self.user_num)+'mnist.csv')
#                 data_size = np.array(data_size['data_size'].tolist())
#
#         self.D = (data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9
#         self.alpha = 0.1
#         self.tau = 1
#         self.C = 20
#         self.communication_time = np.random.uniform(low=10, high=20, size=self.user_num)
#
#         self.BATCH = 5   #todo
#         self.A_UPDATE_STEPS = 5  # origin:5
#         self.C_UPDATE_STEPS = 5
#         self.HAVE_TRAIN = False
#         self.A_LR = 0.00001  # origin:0.00003
#         self.C_LR = 0.00001
#         self.GAMMA = 0.95  # origin: 0.95
#         self.dec = 0.3
#
#         self.EP_MAX = 1000  #todo
#         self.EP_MAX_pre_train = 1000
#
#         self.EP_LEN = 100
#
#
#         if self.user_num == 5:
#             self.delta_max = np.array([1.4359949, 1.02592623, 1.54966248, 1.43532239, 1.4203678])
#
#             Loss = pd.read_csv('loss_mnist_500.csv')
#             Loss = Loss.to_dict()
#             Loss = Loss['1']
#             loss_list = []
#             for i in Loss:
#                 loss_list.append(Loss[i])
#
#
#             self.loss_list = copy.copy(loss_list)
#             num = len(loss_list)
#             buffer = 0
#             profit_increase = []
#             for i in range(0, num):
#                 loss_list[i] = -math.log(loss_list[i])
#
#             for one in loss_list:
#                 profit_increase.append(one - buffer)
#                 buffer = one
#
#             self.acc_increase_list = profit_increase
#
#
#
#         else:
#             self.delta_max = np.random.uniform(low=1, high=2, size=self.user_num)
#             data_info = pd.read_csv('Multi_client_data/' + str(self.user_num) + 'user_' + 'mnist' + '_1_0.005.csv')
#             accuracy_list = data_info['loss'].tolist()
#             num = len(accuracy_list)
#
#             self.loss_list = copy.copy(accuracy_list)
#             for i in range(0, num):
#                 accuracy_list[i] = -math.log(accuracy_list[i])
#
#             buffer = -math.log(3)
#
#             self.acc_increase_list = []
#             for one in accuracy_list:
#                 self.acc_increase_list.append(one - buffer)
#                 buffer = one
#
#         self.amplifier_baseline = np.max(self.delta_max * self.tau * self.C * self.D * self.alpha)
#         self.amplifier_hrl = np.sum(self.delta_max * self.tau * self.C * self.D * self.alpha)
#         self.reducer_baseline = 1
#         self.reducer_hrl = 100
#
#         reducer_pretrain_dict = {
#             5: 1000,
#             10: 500,
#             20: 50,
#             30: 50,
#             40: 50,
#             40: 50,
#             50: 50
#         }
#         self.reducer_pretrain = reducer_pretrain_dict[self.user_num]
#
#
# if __name__ == '__main__':
#
#     np.random.seed(2)
#     configs = Configs('mnist', 50)
#
#     print(configs.delta_max)
#     print(configs.D)
#     print(configs.amplifier_hrl)
#     print(configs.amplifier_baseline)
#     print(configs.communication_time)
#     print(configs.acc_increase_list)
#     # print(configs.D)