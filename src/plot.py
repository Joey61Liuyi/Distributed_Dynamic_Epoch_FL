# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 21:33
# @Author  : LIU YI

import pandas as pd


for j in range(1, 10, 1):

    accumulative_reward = 0
    for i in range(5):
        tep = pd.read_csv('Result_book_of_round_'+str(i)+'.csv')
        data = tep.sample(frac=j/10, random_state=0)
        print(len(data))
        maxreward = data.sort_values('reward').iloc[-1]['reward']
        accumulative_reward += maxreward

    print(j, accumulative_reward)