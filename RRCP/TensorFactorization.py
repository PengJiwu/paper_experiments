__author__ = 'tao'

import numpy as np
import random
import pandas as pd

import Construction

class BasicMetrics():
    def __init__(self):
        a = 0
    def F1_score_Hit_ratio(self, recommend_list, purchas_list):
        user_number = len(recommend_list)
        correct = []
        co_length = []
        re_length = []
        pu_length = []
        p = []
        r = []
        f = []
        hit_number = 0
        for i in range(user_number):
            temp = []
            for j in recommend_list[i]:
                if j in purchas_list[i]:
                    #print 'find one!'
                    temp.append(j)
            if len(temp):
                hit_number = hit_number + 1
            co_length.append(len(temp))
            re_length.append(len(recommend_list[i]))
            pu_length.append(len(purchas_list[i]))
            correct.append(temp)

        for i in range(user_number):
            if re_length[i] != 0 and pu_length[i] != 0:
                p_t = co_length[i] / re_length[i]
                r_t = co_length[i] / pu_length[i]
                p.append(p_t)
                r.append(r_t)
                if p_t != 0 or r_t != 0:
                    f.append(2*p_t*r_t / (p_t+r_t))
                else:
                    f.append(0)
        if user_number != 0:
            hit_tario = hit_number / user_number
        else:
            hit_tario = 0
        print 'Precisions are :' + str(p)
        print 'Recalls are :' + str(r)
        print 'F_1s are :' + str(array(f).mean())
        print 'Hit_ratios are :' + str(hit_tario)

        return p, r, array(f).mean(), hit_tario
    def NDGG_k(self, recommend_list, purchas_list):
        user_number = len(recommend_list)
        u_ndgg = []
        for i in range(user_number):
            temp = 0
            Z_u = 0
            for j in range(len(recommend_list[i])):
                Z_u = Z_u + 1 / log2(j + 2)
                if recommend_list[i][j] in purchas_list[i]:
                    temp = temp + 1 / log2(j + 2)
            if Z_u !=0:
                temp = temp / Z_u
            u_ndgg.append(temp)
        print 'NDGG are :' + str(array(u_ndgg).mean())

        return array(u_ndgg).mean()

class TensorBPR(object):
    def __init__(self,trainSamples, dataModel, implict_dimension_num, num_iters, pf_lambda, sentry, learning_rate, regularization):
        self.D = implict_dimension_num
        self.D_f = implict_dimension_num
        self.learning_rate = learning_rate
        self.bias_regularization = regularization
        self.user_regularization = regularization
        self.positive_price_regularization = regularization
        self.negative_price_regularization = regularization
        self.positive_feature_regularization = regularization
        self.negative_feature_regularization = regularization
        self.update_negative_item_factors = True
        self.pf_lambda = pf_lambda
        self.sentry = sentry

        self.num_iters = num_iters

        self.tensor_train = trainSamples

        self.up_train = []
        self.tensor_train.apply((lambda x: self.up_train.append((x['user'], x['price_ix']))), axis=1)

        self.dataModel = dataModel
        self.num_users = self.dataModel.getUsersNum()
        self.num_prices = self.dataModel.getPriceIxNum()
        self.num_features = self.dataModel.getFeaturesNum()
        print self.num_users, self.num_prices, self.num_features

        self.user_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_users, self.D))
        self.price_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_prices, self.D))
        self.feature_user_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_features, self.D_f))
        self.feature_price_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_features, self.D_f))
    def _gen_samples(self):
        result = []
        self.price_feature_mention = Construction.construct_price_feature(self.tensor_train, self.dataModel)
        for user, user_df in self.price_feature_mention.iteritems():
            for ix in xrange(len(user_df)):
                row = user_df[ix]
                if sum(row):
                    nonzero = row.nonzero()[0]
                    up_result = []
                    while len(up_result) < len(nonzero):
                        sample_i = random.choice(nonzero)

                        sample_j = random.choice(range(self.num_features))
                        while row[sample_i] <= row[sample_j]:
                            sample_j = random.choice(xrange(self.num_features))

                        jx = random.choice(range(self.num_prices))
                        random_time = 0
                        while user_df[ix][sample_i] <= user_df[jx][sample_i] and random_time < self.num_prices:
                            jx = random.choice(range(self.num_prices))
                            random_time += 1

                        if random_time >= self.num_prices:
                            break

                        up_result.append((ix, jx, sample_i, sample_j))
            result.append((user, up_result))
        return result

    def train(self):
        initial = False
        for it in xrange(self.num_iters):
            print 'starting iteration {0}'.format(it)
            samples = self._gen_samples()
            if not initial:
                old_object = self.objectFunction()
                print 'initial loss = {0}'.format(old_object)
                initial = True
            for u, pf_samples in samples:
                self.user_factors[u,:] += self.learning_rate*(-self.user_regularization*self.user_factors[u,:])
                for p1, p2, f1, f2 in pf_samples:
                        x = self.y_uif(u,p1,f1) - self.y_uif(u,p1,f2)
                        z1 = 1 - 1.0/(1.0+np.exp(-x))
                        x = self.y_uif(u, p1, f1) - self.y_uif(u, p2, f1)
                        z2 = 1 - 1.0/(1.0+np.exp(-x))

                        d_ranking = (self.feature_user_factors[f1,:]-self.feature_user_factors[f2,:]) * z1
                        self.user_factors[u,:] += self.learning_rate*d_ranking

                        d_ranking = (self.feature_price_factors[f1,:]-self.feature_price_factors[f2,:]) * z1 + self.pf_lambda * self.feature_price_factors[f1,:] * z2 - self.positive_price_regularization*self.price_factors[p1,:]
                        self.price_factors[p1,:] += self.learning_rate*d_ranking

                        d_ranking = - self.pf_lambda * self.feature_price_factors[f1,:] * z2 - self.negative_price_regularization*self.price_factors[p2,:]
                        self.price_factors[p2,:] += self.learning_rate*d_ranking

                        d_ranking = (self.user_factors[u,:])*z1 - self.positive_feature_regularization*self.feature_user_factors[f1,:]
                        self.feature_user_factors[f1,:] += self.learning_rate*d_ranking

                        d_ranking = (-self.user_factors[u,:])*z1 - self.negative_feature_regularization*self.feature_user_factors[f2,:]
                        self.feature_user_factors[f2,:] += self.learning_rate*d_ranking

                        d_ranking = (self.price_factors[p1,:])*z1 + self.pf_lambda * (self.feature_price_factors[p1,:]-self.feature_price_factors[p2,:]) * z2 - self.positive_feature_regularization*self.feature_price_factors[f1,:]
                        self.feature_price_factors[f1,:] += self.learning_rate*d_ranking

                        d_ranking = (-self.price_factors[p1,:])*z1 - self.negative_feature_regularization*self.feature_price_factors[f2,:]
                        self.feature_price_factors[f2,:] += self.learning_rate*d_ranking
            new_object = self.objectFunction()
            print 'iteration {0}: loss = {1}'.format(it, new_object)
            if new_object < old_object:
                self.learning_rate *= 0.5
            elif new_object - old_object < self.sentry:
                print 'converge!!'
                break
            old_object = new_object


        new_tensor = np.empty((self.num_users, self.num_prices, self.num_features))

        for ix in range(self.num_users):
            for rx in range(self.num_prices):
                tmp = np.dot(self.user_factors[ix,:], self.price_factors[rx,:])
                for cx in range(self.num_features):
                    new_tensor[ix, rx, cx] = np.dot(self.user_factors[ix,:], self.feature_user_factors[cx,:]) + np.dot(self.price_factors[rx,:], self.feature_price_factors[cx,:]) + tmp
        return new_tensor
    def y_uif(self, uid, pid, fid):
        yuif = np.dot(self.user_factors[uid,:], self.feature_user_factors[fid,:])+np.dot(self.price_factors[pid,:], self.feature_price_factors[fid,:])+np.dot(self.user_factors[uid,:], self.price_factors[pid,:])
        return yuif
    def objectFunction(self):
        gap = 0
        #rating_loss = 0
        complexity = 0

        complexity += self.user_regularization * np.sum(np.diag(np.dot(self.user_factors, self.user_factors.transpose())))
        complexity += self.positive_price_regularization * np.sum(np.diag(np.dot(self.price_factors, self.price_factors.transpose())))
        complexity += self.positive_feature_regularization * np.sum(np.diag(np.dot(self.feature_user_factors, self.feature_user_factors.transpose())))
        complexity += self.positive_feature_regularization * np.sum(np.diag(np.dot(self.feature_price_factors, self.feature_price_factors.transpose())))

        for ix, row in self.tensor_train.iterrows():
            u = self.dataModel.getUidByUser(row['user'])
            price_feature = self.price_feature_mention[u]
            p = self.dataModel.getPidByPriceIx(row['price_ix'])

            #score = int(float(line[3]))
            #rating_loss += (score - np.dot(self.user_factors[u],self.item_factors[i])) ** 2

            f1s = [self.dataModel.getFidByFeature(f[0]) for f in eval(row['feature'])]
            f2s = [i for i in range(self.num_features) if i not in f1s]

            for f1 in f1s:
                yuif_1 = self.y_uif(u,p,f1)
                for f2 in f2s:
                    yuif_2 = self.y_uif(u,p,f2)
                    gap += np.log(1/(1.0+np.exp(yuif_2 - yuif_1)))
                #contruction defferent price
                p2s = [p2 for p2 in range(self.num_prices) if price_feature[p, f1] > price_feature[p2, f2]]
                for p2 in p2s:
                    yuif_2 = self.y_uif(u,p2,f2)
                    gap += np.log(1/(1.0+np.exp(yuif_2 - yuif_1)))

        print 1
        return gap - complexity


if __name__ == '__main__':
    '''
    num_factors = 50
    f_tmp = 0
    model = TensorBPR(num_factors)
    num_iters = 10000
    model.train(num_iters)
    '''
    a = pd.read_csv('a').values
    b = pd.read_csv('b').values
    c = pd.read_csv('c').values
    d = pd.read_csv('d').values
    len1 = 689
    len2 = 1036
    len3 = 172
    tensor = np.empty((len1, len2, len3))

    for ix in range(len1):
        for rx in range(len2):
            tmp = np.dot(a[ix,:], b[rx,:])
            for cx in range(len3):
                tensor[ix, rx, cx] = np.dot(a[ix,:], c[cx,:]) + np.dot(b[rx,:], d[cx,:]) + tmp

    data = pd.read_csv('~/Documents/coding/dataset/workplace/filter_phones_train.csv')
    item_feature = Construction.construct_item_feature(data, len2, len3)

    result = np.dot(tensor, item_feature)
    print result




