# -*- coding:utf-8 -*-
import numpy as np
import itertools
from BaseDataModel import BaseDataModel

class MemeryDataModel(BaseDataModel):

    def __init__(self, samples, targets, isRating=False):
        super(MemeryDataModel, self).__init__()
        self.samples = samples
        self.__users = list(set(samples['user']))
        self.__items = list(set(samples['item']))

        self.__buy_time = [[[] for x in self.__items] for _ in self.__users]
        self.__rate = np.zeros((len(self.__users), len(self.__items)))

        for ix, row in samples.iterrows():
            user = row['user']
            item = row['item']
            time = row['time']
            rate = row['rate']

            self.__buy_time[self.getUidByUser(user)][self.getIidByItem(item)].append(time)
            self.__rate[self.getUidByUser(user), self.getIidByItem(item)] = rate

    def getUidByUser(self, user):
        if user not in self.__users:
            return -1
        else:
            uid = np.argwhere(np.array(self.__users) == user)[0][0]
        return uid

    def getIidByItem(self, item):
        if item not in self.__items:
            return -1
        else:
            iid = np.argwhere(np.array(self.__items) == item)[0][0]
        return iid

    def getUserByUid(self, uid):
        return self.__users[uid]

    def getItemByIid(self, iid):
        return self.__items[iid]

    def getUsersNum(self):
        return len(self.__users)

    def getItemsNum(self):
        return len(self.__items)

    def getBuyTimeByUIId(self, userID, itemID):
        return self.__buy_time[userID][itemID]

    def getRateByUIId(self, userID, itemID):
        return self.__rate[userID, itemID]

    def getNonEmptyIidByUid(self, userID):
        nonIid = [x for x in range(self.getItemsNum()) if self.__buy_time[userID][x]]
        return nonIid

    def getBuyTimeBeforeByUIId(self, userID, itemID, time):
        all_time = self.getBuyTimeByUIId(userID, itemID)
        result = [x for x in all_time if x < time]
        # 为后续计算和求导方便，假设每个人至少已经买过一次了
        result.insert(0, 0)
        return result

    def getUserEarliestAndLatestTimeByUid(self, userID):
        item_buy_time = self.__buy_time[userID]
        all_time = list(itertools.chain(*item_buy_time))
        return min(all_time), max(all_time)

    def getGloablLatestTime(self):
        time = []
        for i in xrange(self.getUsersNum()):
            item_buy_time = self.__buy_time[i]
            all_time = list(itertools.chain(*item_buy_time))
            time.append(max(all_time))
        return max(time)