# -*- coding:utf-8 -*-
__author__ = 'tao'

from sklearn.base import BaseEstimator
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from collections import Counter
import random

import MyMemeryDataModel
from Evaluation import *


class UMNU(BaseEstimator):
    def __init__(self, rec_num=5, num_iter=5, sentry=0.1, implict_dim=50, precision_lambda_1=1, gamma_0=1,
                 precision_lambda_2=1, alpha_0=1, precision_lambda_3=1, avg_price=True,beta_1=0.015, beta_2=0.035, beta_3=0.01, neg_pos_ratio=1):
        self.rec_num, self.num_iter, self.sentry, self.implict_dim, self.precision_lambda_1, self.gamma_0, self.precision_lambda_2, self.alpha_0, self.precision_lambda_3, self.avg_price, self.beta_1, self.beta_2, self.beta_3, self.neg_pos_ratio = rec_num, num_iter, sentry, implict_dim, precision_lambda_1, gamma_0, precision_lambda_2, alpha_0, precision_lambda_3, avg_price, beta_1, beta_2, beta_3, neg_pos_ratio

    def _construct(self):
        self.user_factor = np.random.normal(0, 1.0 / self.precision_lambda_1,
                                            (self.dataModel.getUsersNum(), self.implict_dim))
        self.item_factor = np.random.normal(0, 1.0 / self.precision_lambda_1,
                                            (self.dataModel.getItemsNum(), self.implict_dim))
        self.gamma = np.random.normal(self.gamma_0, 1.0 / self.precision_lambda_2, self.dataModel.getItemsNum())

        self.alpha = np.random.normal(self.alpha_0, 1.0 / self.precision_lambda_3, self.dataModel.getUsersNum())

        # target sample only once in initializing
        self.target_samples = self._sample()

    def random_abondon(self, num):
        rand = random.random()
        if rand <= num:
            return True
        else:
            return False

    def _sample(self):
        result = set([])
        user_num = self.dataModel.getUsersNum()
        item_num = self.dataModel.getItemsNum()

        for ix in xrange(user_num):
            iid = self.dataModel.getNonEmptyIidByUid(ix)
            for rx in iid:
                buy_time = self.dataModel.getBuyTimeByUIId(ix, rx)
                for time in buy_time:
                    result.add((ix, rx, time, True))

        # 用户不可以random，必须每个用户都有负例，防止user_factor太大
        # 物品也不可以random，用户购买的同一商品不能再有负例，模型缺陷！！
        # 而且每个有交易记录的物品也必须有负的例子，防止 gamma 太大
        pos_obj = set([(x[0], x[1]) for x in result])
        valid_u = [[(ux, ix) for ix in xrange(item_num) if (ux, ix) not in pos_obj] for ux in xrange(user_num)]
        valid_i = [[(ux, ix) for ux in xrange(user_num) if (ux, ix) not in pos_obj] for ix in xrange(item_num)]

        neg_obj = []
        neg_sample_len = len(result) * self.neg_pos_ratio
        while len(neg_obj) < neg_sample_len:
            for ui_list in valid_u:
                ui_ix = random.choice(len(ui_list))
                neg_obj.append(ui_list[ui_ix])
            for ui_list in valid_i:
                ui_ix = random.choice(len(ui_list))
                neg_obj.append(ui_list[ui_ix])

        neg_result = []
        for ux, rx in neg_obj:
            time = 1
            neg_result.append((ux, rx, time, False))

        result = list(result)
        for ui in neg_result:
            ix = random.randint(len(result))
            result.insert(ix, ui)
        return result

    def _margial_ratio(self, user, item, time):
        before_buy = len(self.dataModel.getBuyTimeBeforeByUIId(user, item, time))
        margin = (before_buy + 1) ** self.gamma[item] - before_buy ** self.gamma[item]

        return margin

    def _rectifier(self, uid, iid):
        u = self.dataModel.getUserByUid(uid)
        i = self.dataModel.getItemByIid(iid)
        if self.avg_price:
            price_in_train = self.user_buy_avg_price[u]
        else:
            price_in_train = self.user_max_price[u]
        price = self.item_price[i]

        if price > price_in_train:
            return (0.0 + price - price_in_train) / price_in_train
        else:
            return 0.0

    def _marginal_net_utility(self, user, item, time):
        margin = self._margial_ratio(user, item, time)
        price_param = np.exp(-self.alpha[user] * self._rectifier(user, item))
        utility = price_param * np.dot(self.user_factor[user], self.item_factor[item]) * margin - self.item_price[self.dataModel.getItemByIid(item)]
        return utility

    def _target_value(self):
        result = 0.0
        first_true = False
        first_false = False
        for user, item, time, is_bought in self.target_samples:
            if is_bought:
                value = self._marginal_net_utility(user, item, time)
                if not first_true:
                    # print user, item, time, is_bought, value
                    first_true = True
            else:
                value = -self._marginal_net_utility(user, item, time)
                if not first_false:
                    # print user, item, time, is_bought, value
                    first_false = True

            result += value

        return result

    def fit(self, trainSamples, trainTargets):
        self.trainDataFrame = pd.DataFrame(trainSamples, columns=['user', 'item', 'title', 'price', 'rate', 'time'])
        self.item_price = self.trainDataFrame.set_index('item')['price'].to_dict()
        tmpDataFrame = self.trainDataFrame.groupby('user').mean()
        self.user_buy_avg_price = tmpDataFrame['price'].to_dict()
        tmpDataFrame = self.trainDataFrame.groupby(['user']).min()
        self.user_min_price = tmpDataFrame['price'].to_dict()
        tmpDataFrame = self.trainDataFrame.groupby(['user']).max()
        self.user_max_price = tmpDataFrame['price'].to_dict()

        self.dataModel = MyMemeryDataModel.MemeryDataModel(self.trainDataFrame, trainTargets)

        self._construct()

        initial = False
        origin_beta_1 = self.beta_1
        origin_beta_2 = self.beta_2
        origin_beta_3 = self.beta_3

        for it in xrange(self.num_iter):
            # a = max(self.gamma)
            # print a, list(self.gamma).count(a)
            # print 'starting iteration {0}'.format(it)
            if not initial:
                old_target = self._target_value()
                # print 'initial loss = {0}'.format(old_target)
                initial = True

            samples = self._sample()

            for user, item, time, is_bought in samples:
                margin = self._margial_ratio(user, item, time)
                rect = self._rectifier(user, item)
                price_param = np.exp(-self.alpha[user] * rect)
                partial_p_u = self.item_factor[item] * margin * price_param
                partial_q_i = self.user_factor[user] * margin * price_param

                pq = np.dot(self.user_factor[user], self.item_factor[item])
                before_buy = len(self.dataModel.getBuyTimeBeforeByUIId(user, item, time))
                margin_gamma_i = (before_buy + 1) ** self.gamma[item] * np.log(before_buy + 1) - before_buy ** \
                                                                                                 self.gamma[
                                                                                                     item] * np.log(
                    before_buy)
                partial_gamma_i = price_param * pq * margin_gamma_i

                price = self.item_price[self.dataModel.getItemByIid(item)]
                if self.avg_price:
                    u_price = self.user_buy_avg_price[self.dataModel.getUserByUid(user)]
                else:
                    u_price = self.user_max_price[self.dataModel.getUserByUid(user)]
                if price > u_price:
                    partial_alpha_u = -rect * price_param * pq * margin
                else:
                    partial_alpha_u = 0.0

                if not is_bought:
                    partial_p_u = -partial_p_u
                    partial_q_i = -partial_q_i
                    partial_gamma_i = -partial_gamma_i
                    partial_alpha_u = -partial_alpha_u

                self.user_factor[user] += self.beta_1 * partial_p_u
                self.item_factor[item] += self.beta_1 * partial_q_i
                self.gamma[item] += self.beta_2 * partial_gamma_i
                if self.gamma[item] > 1:
                    self.gamma[item] = 1
                self.alpha[user] += self.beta_3 * partial_alpha_u
                if self.alpha[user] < 0:
                    self.alpha[user] = 0

            new_target = self._target_value()
            # print 'iteration {0}: loss = {1}'.format(it, new_target)
            # check and adapt learning rate
            if new_target < old_target:
                self.beta_1 *= 0.5
                self.beta_2 *= 0.5
                self.beta_3 *= 0.5
            elif new_target - old_target < self.sentry:
                print 'converge!!'
                break
            else:
                self.beta_1 *= 0.999
                self.beta_2 *= 0.999
                self.beta_3 *= 0.999
            old_target = new_target
        self.beta_1 = origin_beta_1
        self.beta_2 = origin_beta_2
        self.beta_3 = origin_beta_3

        # self.user_factor = np.ones((self.dataModel.getUsersNum(), self.implict_dim))
        # self.item_factor = np.ones((self.dataModel.getItemsNum(), self.implict_dim))
        # self.gamma = [1 for _ in xrange(self.dataModel.getItemsNum())]
        # for k in self.item_price.keys():
        # self.item_price[k] = 0

    def recommend(self, u, time):
        uid = self.dataModel.getUidByUser(u)
        utility = [self._marginal_net_utility(uid, i, time) for i in xrange(self.dataModel.getItemsNum())]
        recommend_items = sorted(xrange(len(utility)), key=lambda x: utility[x], reverse=True)
        real_items = [self.dataModel.getItemByIid(x) for x in recommend_items]
        min_price = self.user_min_price[u]
        max_price = self.user_max_price[u]
        #real_items = [x for x in real_items if self.item_price[x] >= min_price and self.item_price[x] <= max_price]
        return real_items[:self.rec_num]

    def score(self, testSamples, trueLabels=None):
        testSamples = pd.DataFrame(testSamples, columns=['user', 'item', 'title', 'price', 'rate', 'time'])
        trueList = []
        recommendList = []
        user_unique = list(set(testSamples['user']))
        #print 'user length:', len(user_unique)
        count = 0
        for u in user_unique:
            true_time = set(testSamples[testSamples.user == u]['time'])
            for time in true_time:
                tmp_samples = testSamples[testSamples.user == u]
                true_item = list(set(tmp_samples[tmp_samples.time == time]['item']))
                trueList.append(true_item)
                pre = self.recommend(u, time)

                if max([self.item_price[x] for x in pre]) > self.user_max_price[u] and max([self.item_price[x] for x in pre]) > max([self.item_price[x] for x in true_item]):
                    count += 1
                #if max([self.item_price[x] for x in pre]) > self.user_max_price[u]:
                    #print [self._marginal_net_utility(self.dataModel.getUidByUser(u), self.dataModel.getIidByItem(x), time) for x in pre]
                #print self.user_max_price[u], max([self.item_price[x] for x in true_item]), max([self.item_price[x] for x in pre])
                # print self.user_min_price[u], self.user_max_price[u]
                recommendList.append(pre)
        print count
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'UMNU result:' + '(' + str(self.get_params()) + '):\t' + str((result)['F1'])
        return (result)['F1']


class CustomCV(object):
    def __init__(self, ids, n_folds):
        """Pass an array of phenomenon ids"""
        self.df = ids
        self.n_folds = n_folds

    def __iter__(self):
        for i in range(self.n_folds):
            sample_size = random.uniform(0.7, 0.9)
            df = self.df.sort(['time'])

            users_all = df['user']
            users_set = list(set(users_all))
            users_records_num = [list(users_all).count(i) for i in users_set]
            train_num = [int(i * sample_size) for i in users_records_num]
            user_num = np.zeros(len(users_set))
            train = []
            test = []
            length = len(df)
            # filtering
            valid_items = set([])
            for ix in range(length):
                uid = df.ix[ix, 'user']
                iid = df.ix[ix, 'item']
                # print uid, uid in users_set, len(user_num), len(train_num)
                if user_num[uid] < train_num[uid]:
                    user_num[uid] += 1
                    valid_items.add(iid)

            user_num = np.zeros(len(users_set))
            for ix in range(length):
                uid = df.ix[ix, 'user']
                iid = df.ix[ix, 'item']
                if user_num[uid] < train_num[uid]:
                    user_num[uid] += 1
                    train.append(ix)
                elif iid in valid_items:
                    test.append(ix)

            yield np.array(train), np.array(test)

    def __len__(self):
        return self.n_folds


if __name__ == '__main__':
    df = pd.read_csv('../preprocess/phonesu5i5_format.csv')
    # df = pd.read_csv('~/Documents/coding/dataset/workplace/phonesu5i5_format.csv')
    data = df
    targets = df['rate']

    umnu = UMNU()
    parameters = {'rec_num': [5], 'num_iter': [2000], 'sentry': [20, 10, 1, 0.1], 'implict_dim': [150],
                  'precision_lambda_1': [0.25, 0.64, 1, 4, 9, 16, 25], 'gamma_0': [0.3, 0.5, 0.8, 1, 2], 'precision_lambda_2': [0.25, 0.64, 1, 4, 9, 16, 25],
                  'alpha_0':[0.1, 0.2, 0.3, 0.5, 0.8, 1, 2], 'precision_lambda_3':[0.25, 0.64, 1, 4, 9, 16, 25], 'beta_1': [0.001, 0.003, 0.008, 0.01, 0.03, 0.1, 0.3], 'beta_2': [0.001, 0.003, 0.008, 0.01, 0.03, 0.1, 0.3],
                  'beta_3':[0.001, 0.003, 0.008, 0.01, 0.03, 0.1, 0.3], 'avg_price':[True, False],
                  'neg_pos_ratio': [0.3, 0.5, 0.8, 1, 10]}
    my_cv = CustomCV(data, 2)
    clf = grid_search.GridSearchCV(umnu, parameters, cv=my_cv, n_jobs=1)
    clf.fit(data, targets)
    #print(clf.grid_scores_)
    print clf.best_score_, clf.best_params_
