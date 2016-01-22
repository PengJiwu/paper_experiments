#-*- coding:utf-8 -*-
__author__ = 'tao'

from sklearn.base import BaseEstimator
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from collections import Counter

import MyMemeryDataModel
from Evaluation import *

class UMNU4Time(BaseEstimator):
    def __init__(self, rec_num=5, num_iter=5, sentry=0.1, implict_dim=50, precision_lambda_1=1, gamma_0=1, precision_lambda_2=1, beta_1=0.015, beta_2=0.035, neg_pos_ratio=1):
        self.rec_num, self.num_iter, self.sentry, self.implict_dim, self.precision_lambda_1, self.gamma_0, self.precision_lambda_2, self.beta_1, self.beta_2, self.neg_pos_ratio = rec_num, num_iter, sentry, implict_dim, precision_lambda_1, gamma_0, precision_lambda_2, beta_1, beta_2, neg_pos_ratio

    def _construct(self):
        self.user_factor = np.random.normal(0, 1.0 / self.precision_lambda_1,
                                            (self.dataModel.getUsersNum(), self.implict_dim))
        self.item_factor = np.random.normal(0, 1.0 / self.precision_lambda_1,
                                            (self.dataModel.getItemsNum(), self.implict_dim))
        self.gamma = np.random.normal(self.gamma_0, 1.0 / self.precision_lambda_2, self.dataModel.getItemsNum())

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
            iid = self.dataModel.getNonEmptyIidByUid(ix)
            for rx in iid:
                buy_time = self.dataModel.getBuyTimeByUIId(ix, rx)
                for time in buy_time:
                    result.add((ix, rx, time, True))

        # 用户不可以random，必须每个用户都有负例，防止user_factor太大
        # 物品也不可以random，用户购买的同一商品不能再有负例，模型缺陷！！在4Time里可以有
        # 而且每个有交易记录的物品也必须有负的例子，防止 gamma 太大
        pos_obj = set([(x[0], x[1]) for x in result])
        valid_u = [[(ux, ix) for ix in xrange(item_num)] for ux in xrange(user_num)]
        valid_i = [[(ux, ix) for ux in xrange(user_num)] for ix in xrange(item_num)]

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
        global_latest_time = self.dataModel.getGloablLatestTime()
        for ux, rx in neg_obj:
            earliest_time, foo = self.dataModel.getUserEarliestAndLatestTimeByUid(ux)
            time = random.randint(earliest_time, global_latest_time)
            neg_result.append((ux, rx, time, False))

        result = list(result)
        for ui in neg_result:
            ix = random.randint(len(result))
            result.insert(ix, ui)
        return result

    def _margial_ratio(self, user, item, time):
        times = self.dataModel.getBuyTimeBeforeByUIId(user, item, time)
        before_buy = len(times)
        if before_buy == 1:
            margin = 1
        else:
            time_span = time - times[-1]
            item_avg_time_span = self.item_time.get(self.dataModel.getItemByIid(item), self.item_time['global_avg'])
            time_para = (np.exp(1) - 1) * before_buy / (1 + np.exp(-time_span / item_avg_time_span + self.gamma[item]))
            margin = np.log(before_buy + time_para) - np.log(before_buy)

        return margin

    def _marginal_net_utility(self, user, item, time):
        margin = self._margial_ratio(user, item, time)
        utility = np.dot(self.user_factor[user], self.item_factor[item]) * margin - self.item_price[
            self.dataModel.getItemByIid(item)]
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

    def _gen_avg_item_buy_time_apply(self, row, item_times):
        id = (row['user'], row['item'])
        time = row['time']
        if id in item_times:
            item_times[id].add(time)
        else:
            item_times[id] = set([time])

    def gen_avg_item_buy_time(self):
        item_times = {}
        self.trainDataFrame.apply((lambda x: self._gen_avg_item_buy_time_apply(x, item_times)), axis=1)

        tmp_result = {}
        for ix, value in item_times.iteritems():
            #单个用户的平均时间间隔
            if len(value) > 1:
                value = sorted(list(value), reverse=True)
                sum = 0.0
                for i in range(len(value)-1):
                    sum += value[i] - value[i+1]
                avg = sum / (len(value) - 1)

                #多个用户的
                item = int(ix[1])
                if item in tmp_result:
                    user_num, avg_time = tmp_result[item]
                    new_avg = (user_num * avg_time + avg) / (user_num + 1)
                    tmp_result[item] = (user_num+1, new_avg)
                else:
                    tmp_result[item] = (1, avg)
        result = {}
        sum = 0.0
        for ix, value in tmp_result.iteritems():
            result[ix] = value[1]
            sum += value[1]
        global_avg = sum / len(tmp_result)
        result['global_avg'] = global_avg
        return result

    def fit(self, trainSamples, trainTargets):
        self.trainDataFrame = pd.DataFrame(trainSamples, columns = ['user', 'item', 'title', 'price', 'rate', 'time'])
        self.item_price = self.trainDataFrame.set_index('item')['price'].to_dict()
        self.item_time = self.gen_avg_item_buy_time()
        self.dataModel = MyMemeryDataModel.MemeryDataModel(self.trainDataFrame, trainTargets)

        self._construct()

        initial = False
        origin_beta_1 = self.beta_1
        origin_beta_2 = self.beta_2

        for it in xrange(self.num_iter):
            # a = max(self.gamma)
            # print a, list(self.gamma).count(a)
            #print 'starting iteration {0}'.format(it)
            if not initial:
                old_target = self._target_value()
                #print 'initial loss = {0}'.format(old_target)
                initial = True

            samples = self._sample()

            for user, item, time, is_bought in samples:
                margin = self._margial_ratio(user, item, time)
                partial_p_u = self.item_factor[item] * margin
                partial_q_i = self.user_factor[user] * margin

                pq = np.dot(self.user_factor[user], self.item_factor[item])
                before_buy = len(self.dataModel.getBuyTimeBeforeByUIId(user, item, time))
                times = self.dataModel.getBuyTimeBeforeByUIId(user, item, time)
                if len(times) == 1:
                    partial_gamma_i = 0
                else:
                    time_span = time - times[-1]
                    item_avg_time_span = self.item_time.get(self.dataModel.getItemByIid(item), self.item_time['global_avg'])
                    time_para = (np.exp(1) - 1) * before_buy / (1 + np.exp(-time_span / item_avg_time_span + self.gamma[item]))
                    partial_gamma_i = time_para * (1 - 1/(1+(1 + np.exp(-time_span / item_avg_time_span + self.gamma[item])))) / (before_buy + time_para) * pq

                if not is_bought:
                    partial_p_u = -partial_p_u
                    partial_q_i = -partial_q_i
                    partial_gamma_i = -partial_gamma_i

                self.user_factor[user] += self.beta_1 * partial_p_u
                self.item_factor[item] += self.beta_1 * partial_q_i
                self.gamma[item] += self.beta_2 * partial_gamma_i


            new_target = self._target_value()
            #print 'iteration {0}: loss = {1}'.format(it, new_target)
            #check and adapt learning rate
            if new_target < old_target:
                self.beta_1 *= 0.5
                self.beta_2 *= 0.5
            elif new_target - old_target < self.sentry:
                print 'converge!!'
                break
            else:
                self.beta_1 *= 0.99
                self.beta_2 *= 0.99
            old_target = new_target
        self.beta_1 = origin_beta_1
        self.beta_2 = origin_beta_2


    def recommend(self, u):
        u = self.dataModel.getUidByUser(u)
        min_time, max_time = self.dataModel.getUserEarliestAndLatestTimeByUid(u)
        now_time = max_time + 1
        utility = [self._marginal_net_utility(u, i, now_time) for i in xrange(self.dataModel.getItemsNum())]
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
            #print pre, true
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
                # print uid, uid in users_set, len(user_num), len(train_num)
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
    df = pd.read_csv('../preprocess/phonesu5i5_format.csv')
    # df = pd.read_csv('~/Documents/coding/dataset/workplace/phonesu5i5_format.csv')
    data = df.values
    targets = [i[4] for i in data]

    umnu = UMNU4Time()
    parameters = {'rec_num': [5], 'num_iter': [1000], 'sentry': [20, 10, 5, 1], 'implict_dim': [25, 50, 75, 100, 150],
                  'precision_lambda_1': [0.5, 1, 2, 4, 10], 'gamma_0': [0.3, 0.5, 0.7], 'precision_lambda_2': [4, 9, 16, 25, 36],
                  'beta_1': [0.003, 0.008, 0.01, 0.012], 'beta_2': [0.003, 0.008, 0.01, 0.012],
                  'neg_pos_ratio': [0.5, 0.8, 1]}
    my_cv = CustomCV(data, 1)
    clf = grid_search.GridSearchCV(umnu, parameters, cv=my_cv, n_jobs=6)
    clf.fit(data, targets)
    print(clf.grid_scores_)
    print clf.best_score_, clf.best_params_



