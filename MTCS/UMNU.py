#-*- coding:utf-8 -*-
__author__ = 'tao'

from sklearn.base import BaseEstimator
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold
import numpy as np

import MyMemeryDataModel
from Evaluation import *


class UMNU(BaseEstimator):
    def __init__(self, rec_num=5, num_iter=5, sentry=0.1, implict_dim=50, precision_lambda_1=1, gamma_0=1, precision_lambda_2=1, beta_1=0.015, beta_2=0.035, neg_pos_ratio=1):
        self.rec_num, self.num_iter, self.sentry, self.implict_dim, self.precision_lambda_1, self.gamma_0, self.precision_lambda_2, self.beta_1, self.beta_2, self.neg_pos_ratio = rec_num, num_iter, sentry, implict_dim, precision_lambda_1, gamma_0, precision_lambda_2, beta_1, beta_2, neg_pos_ratio

    def _construct(self):
        self.user_factor = np.random.normal(0, 1/np.sqrt(self.precision_lambda_1), (self.dataModel.getUsersNum(), self.implict_dim))
        self.item_factor = np.random.normal(0, 1/np.sqrt(self.precision_lambda_1), (self.dataModel.getItemsNum(), self.implict_dim))
        self.gamma = np.random.normal(self.gamma_0, 1/np.sqrt(self.precision_lambda_2), self.dataModel.getItemsNum())

        #target sample only once in initializing
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
            if self.random_abondon(0.5):
                #并不是全部正例都囊括在内，以0.1的概率抛弃
                continue
            iid = self.dataModel.getNonEmptyIidByUid(ix)
            for rx in iid:
                buy_time = self.dataModel.getBuyTimeByUIId(ix, rx)
                for time in buy_time:
                    if self.random_abondon(0.5):
                        #并不是全部正例都囊括在内，以0.1的概率抛弃
                        continue
                    result.add((ix, rx, time, True))
        pos_num = len(result)
        wanted_num = (self.neg_pos_ratio + 1) * pos_num
        while len(result) < wanted_num:
            ix = random.randint(user_num)
            rx = random.randint(item_num)
            #sample time from min to max
            min_time, max_time = self.dataModel.getUserEarliestAndLatestTimeByUid(ix)
            if min_time == max_time:
                if self.random_abondon(0.5):
                    time = min_time - random.randint(3600 * 24 * 365)
                else:
                    time = min_time + random.randint(3600 * 24 * 365)
            else:
                time = random.randint(min_time, max_time)
            result.add((ix, rx, time, False))
        return result

    def _margin_ratio(self, user, item, time):
        before_buy = len(self.dataModel.getBuyTimeBeforeByUIId(user, item, time))
        if before_buy == 0:
            margin = 1
        else:
            margin = (before_buy + 1) ** self.gamma[item] - before_buy ** self.gamma[item]
        return margin

    def _marginal_net_utility(self, user, item, time, is_bought):
        before_buy = len(self.dataModel.getBuyTimeBeforeByUIId(user, item, time))
        if before_buy == 0:
            margin = 1
        else:
            margin = (before_buy + 1) ** self.gamma[item] - before_buy ** self.gamma[item]
        utility = self._marginal_net_utility_from_margin_ratio(user, item, margin, is_bought)
        return utility

    def _marginal_net_utility_from_margin_ratio(self, user, item, margin_ratio, is_bought):
        v_uit = np.dot(self.user_factor[user], self.item_factor[item]) * margin_ratio - self.item_price[self.dataModel.getItemByIid(item)]
        r_uit = 1 if is_bought else -1
        if np.isinf(np.exp(-v_uit * r_uit)):
            utility = -v_uit * r_uit
        else:
            utility = np.log(1 + np.exp(-v_uit * r_uit))
        return utility

    def _target_value(self):
        pre_user_factor = self.precision_lambda_1 * np.sum([np.linalg.norm(x) for x in self.user_factor]) / 2
        pre_item_factor = self.precision_lambda_1 * np.sum([np.linalg.norm(x) for x in self.item_factor]) / 2
        pre_lambda = self.precision_lambda_2 * np.sum([(x - self.gamma_0) ** 2 for x in self.gamma]) / 2

        sum_utility = 0.0
        for user, item, time, is_bought in self.target_samples:
            '''
            if np.isinf(sum_utility + self._marginal_net_utility_from_margin_ratio(user, item, margin_ratio, is_bought)) or np.isnan(sum_utility + self._marginal_net_utility_from_margin_ratio(user, item, margin_ratio, is_bought)):
                b = self._margin_ratio(user, item, time)
                a = sum_utility + self._marginal_net_utility_from_margin_ratio(user, item, b, is_bought)
            '''
            margin_ratio = self._margin_ratio(user, item, time)
            utility = self._marginal_net_utility_from_margin_ratio(user, item, margin_ratio, is_bought)
            if np.isnan(utility) or np.isinf(utility):
                continue
            sum_utility += utility

        result = pre_user_factor + pre_item_factor + pre_lambda + sum_utility
        return result

    def fit(self, trainSamples, trainTargets):
        self.trainDataFrame = pd.DataFrame(trainSamples, columns = ['user', 'item', 'title', 'price', 'rate', 'time'])
        self.item_price = self.trainDataFrame.set_index('item')['price'].to_dict()
        for k in self.item_price.keys():
            self.item_price[k] = 0
        self.dataModel = MyMemeryDataModel.MemeryDataModel(self.trainDataFrame, trainTargets)

        self._construct()

        initial = False
        origin_beta_1 = self.beta_1
        origin_beta_2 = self.beta_2
        for it in xrange(self.num_iter):
            #print max(self.gamma)
            #print 'starting iteration {0}'.format(it)
            if not initial:
                old_target = self._target_value()
                print 'initial loss = {0}'.format(old_target)
                initial = True
            samples = self._sample()

            for user, item, time, is_bought in samples:
                #symbol like paper
                before_buy = len(self.dataModel.getBuyTimeBeforeByUIId(user, item, time))
                if before_buy == 0:
                    d_uit = 1
                else:
                    d_uit = (before_buy + 1) ** self.gamma[item] - before_buy ** self.gamma[item]
                v_uit = self._marginal_net_utility_from_margin_ratio(user, item, d_uit, is_bought)
                r_uit = 1 if is_bought else -1
                if np.isinf(np.exp(-v_uit * r_uit)) or np.isnan(np.exp(-v_uit * r_uit)):
                    g_uit = -r_uit
                else:
                    g_uit = np.exp(-v_uit * r_uit) / (1 + np.exp(-v_uit * r_uit)) * (-r_uit)

                partial_q_i = self.precision_lambda_1 * self.item_factor[item] + g_uit * self.user_factor[user] * d_uit
                partial_p_u = self.precision_lambda_1 * self.user_factor[user] + g_uit * self.item_factor[item] * d_uit
                if before_buy == 0:
                    partial_gamma_i = self.precision_lambda_2 * self.gamma[item] + g_uit * (np.dot(self.user_factor[user], self.item_factor[item]) *
                                                                                        ((before_buy + 1) ** self.gamma[item] * np.log(before_buy + 1)))
                else:
                    partial_gamma_i = self.precision_lambda_2 * self.gamma[item] + g_uit * (np.dot(self.user_factor[user], self.item_factor[item]) *
                                                                                        ((before_buy + 1) ** self.gamma[item] * np.log(before_buy + 1) -
                                                                                        before_buy ** self.gamma[item] * np.log(before_buy)))
                self.user_factor[user] -= self.beta_1 * partial_p_u
                self.item_factor[item] -= self.beta_1 * partial_q_i
                self.gamma[item] -= self.beta_2 * partial_gamma_i
                '''
                print 'partial p_u', max(self.beta_1 * partial_p_u)#, self.beta_1 * partial_p_u
                print 'partial q_u', max(self.beta_1 * partial_q_i)#, self.beta_1 * partial_q_i
                print 'partial gamma', self.beta_2 * partial_gamma_i
                '''
                #debug
                #print 'loss = {0} after update a sample'.format(self._target_value())

            new_target = self._target_value()
            print 'iteration {0}: loss = {1}'.format(it, new_target)
            #check and adapt learning rate
            if old_target < new_target:
                self.beta_1 *= 0.5
                self.beta_2 *= 0.5
            elif old_target - new_target < self.sentry:
                print 'converge!!'
                break
            else:
                self.beta_1 *= 0.99
                self.beta_2 *= 0.99
            old_target = new_target
        self.beta_1 = origin_beta_1
        self.beta_2 = origin_beta_2

        #self.user_factor = np.ones((self.dataModel.getUsersNum(), self.implict_dim))
        #self.item_factor = np.ones((self.dataModel.getItemsNum(), self.implict_dim))
        #self.gamma = [1 for _ in xrange(self.dataModel.getItemsNum())]
        #for k in self.item_price.keys():
            #self.item_price[k] = 0

    def recommend(self, u):
        u = self.dataModel.getUidByUser(u)
        min_time, max_time = self.dataModel.getUserEarliestAndLatestTimeByUid(u)
        now_time = max_time + 1
        utility = [self._marginal_net_utility(u, i, now_time, True) for i in xrange(self.dataModel.getItemsNum())]
        #utility = [np.dot(self.user_factor[u], self.item_factor[i]) for i in xrange(self.dataModel.getItemsNum())]
        recommend_items = sorted(xrange(len(utility)), key=lambda x: utility[x], reverse=True)[:self.rec_num]
        real_items = [self.dataModel.getItemByIid(x) for x in recommend_items]
        return real_items

    def score(self, testSamples, trueLabels=None):
        testSamples = pd.DataFrame(testSamples, columns = ['user', 'item', 'title', 'price', 'rate', 'time'])
        trueList = []
        recommendList= []
        user_unique = list(set(testSamples['user']))
        for u in user_unique:
            true = list(set(testSamples[testSamples.user == u]['item']))
            trueList.append(true)
            pre = self.recommend(u)
            print pre, true
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'UMNU result:'+'('+str(self.get_params())+'):\t' + str((result)['F1'])
        return (result)['F1']

class CustomCV(object):
    def __init__(self, ids, n_folds):
        """Pass an array of phenomenon ids"""
        self.df = pd.DataFrame(ids, columns = ['user', 'item', 'title', 'price', 'rate', 'time'])
        self.n_folds = n_folds

    def __iter__(self):
        for i in range(self.n_folds):
            df = self.df.sort(['time'])

            users_all = np.array(df.values)[:,0]
            users_set = list(set(users_all))
            users_records_num = [list(users_all).count(i) for i in users_set]
            train_num = [int(i*0.8) for i in users_records_num]
            user_num = np.zeros(len(users_set))
            train = []
            test = []
            lines = df.values
            #filtering
            valid_items = set([])
            for line in df.values:
                uid = int(line[0])
                if user_num[uid] < train_num[uid]:
                    user_num[uid] += 1
                    valid_items.add(line[1])

            user_num = np.zeros(len(users_set))
            for ix in xrange(len(lines)):
                line = lines[ix]
                uid = int(line[0])
                if user_num[uid] < train_num[uid]:
                    user_num[uid] += 1
                    train.append(ix)
                elif line[1] in valid_items:
                    test.append(ix)

            yield np.array(train), np.array(test)
    def __len__(self):
        return self.n_folds

if __name__ == '__main__':
    #df = pd.read_csv('../preprocess/Clothing20_train.csv')
    df = pd.read_csv('~/Documents/coding/dataset/workplace/phones3_train.csv')
    data = df.values
    targets = [i[4] for i in data]

    umnu = UMNU()
    parameters = {'rec_num':[5], 'num_iter':[100, 200], 'sentry':[1, 0.1], 'implict_dim':[50, 100], 'precision_lambda_1':[1, 0.1], 'gamma_0':[0.5, 1], 'precision_lambda_2':[1, 0.1], 'beta_1':[1, 0.1], 'beta_2':[1, 0.1], 'neg_pos_ratio':[1,5]}
    my_cv = CustomCV(data, 1)
    clf = grid_search.GridSearchCV(umnu, parameters,cv=my_cv)
    clf.fit(data, targets)
    print(clf.grid_scores_)
    print clf.best_score_, clf.best_params_



