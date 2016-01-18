from __future__ import division
import numpy as np
from math import exp
import random
#import SparseTensor as st
import pandas as pd
import math
import datetime

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
    def __init__(self,D):
        self.D = D
        self.D_f = D
        self.learning_rate = 0.001
        self.bias_regularization = 0.0003
        self.user_regularization = 0.0003
        self.item_regularization = 0.0003
        self.positive_feature_regularization = 0.0003
        self.negative_feature_regularization = 0.0003
        self.update_negative_item_factors = True
        self.file_users = 'tinytest/uid.csv'
        self.file_items = 'tinytest/iid.csv'
        self.file_features = 'tinytest/fid.csv'
        self.trainfile = 'tinytest/unifiedtrain.csv'
        self.testfile = 'tinytest/unifiedtest.csv'
        #self.unify_testfile='yelpv2/6/test_data.csv'
        #self.result_rating = 'yelpv2/6/feature_rating.csv'

        self.alpha = 0.7

        self.tensor_users = pd.read_csv(self.file_users)
        self.tensor_items = pd.read_csv(self.file_items)
        self.tensor_features = pd.read_csv(self.file_features)
        self.tensor_train = pd.read_csv(self.trainfile, dtype='str')
        self.tensor_test = pd.read_csv(self.testfile, dtype='str')
        #self.tensor_unifytest = pd.read_csv(self.unify_testfile, dtype='str')
        #self.ratings = pd.read_csv(self.result_rating, dtype='str')

        self.ui_train = []
        for line in self.tensor_train.values:
            user = int(line[1].split('@')[0])
            item = int(line[1].split('@')[1])
            self.ui_train.append([user,item])
        self.ui_test = []
        for line in self.tensor_test.values:
            user = int(line[1].split('@')[0])
            item = int(line[1].split('@')[1])
            self.ui_test.append([user,item])

        self.num_users = len(self.tensor_users.values)
        self.num_items = len(self.tensor_items.values)
        self.num_features = len(self.tensor_features.values)
        print self.num_users, self.num_items, self.num_features
        self.ave = 3

        self.item_bias = np.zeros(self.num_items)
        self.user_bias = np.zeros(self.num_users)
        self.user_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_users, self.D))
        self.item_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_items, self.D))
        self.feature_user_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_features, self.D_f))
        self.feature_item_factors = 1/(10*np.sqrt(self.D))*np.random.random_sample((self.num_features, self.D_f))
    def sample_negative_feature(self,features_list):
        j = random.randint(0, self.num_features-1)
        while j in features_list:
            j = random.randint(0, self.num_features-1)
        return j
    def train(self, num_iters):
        old_loss = self.loss_hybrid()
        print 'initial loss = {0}'.format(self.loss_hybrid())
        for it in xrange(num_iters):
            print 'starting iteration {0}'.format(it)
            for line in self.tensor_train.values:
                u = int(line[1].split('@')[0])
                i = int(line[1].split('@')[1])
                f1s = [f[0] for f in eval(line[2])]
                f2s = []
                for _ in range(len(f1s)):
                    f2s.append(self.sample_negative_feature(f1s))
                score = int(float(line[3]))
                score_p = score - np.dot(self.user_factors[u], self.item_factors[i])
                temp_u = self.user_factors[u,:]
                self.user_bias[u] += self.learning_rate*(self.alpha*score_p - self.bias_regularization*self.user_bias[u])
                self.item_bias[i] += self.learning_rate*(self.alpha*score_p - self.bias_regularization*self.item_bias[i])
                self.user_factors[u,:] += self.learning_rate*(self.alpha*score_p*self.item_factors[i,:]-self.user_regularization*self.user_factors[u,:])
                self.item_factors[i,:] += self.learning_rate*(self.alpha*score_p*temp_u-self.item_regularization*self.item_factors[i,:])
                for f1 in f1s:
                    for f2 in f2s:

                        x = self.y_uif(u,i,f1) - self.y_uif(u,i,f2)
                        z = 1.0/(1.0+exp(x))
                        d_ranking = (self.feature_user_factors[f1,:]-self.feature_user_factors[f2,:])*z
                        d = (1-self.alpha)*d_ranking
                        self.user_factors[u,:] += self.learning_rate*d

                        d_ranking = (self.feature_item_factors[f1,:]-self.feature_item_factors[f2,:])*z
                        d = (1-self.alpha)*d_ranking
                        self.item_factors[i,:] += self.learning_rate*d

                        d_ranking = (self.user_factors[u,:])*z
                        d = (1-self.alpha)*d_ranking - self.positive_feature_regularization*self.feature_user_factors[f1,:]
                        self.feature_user_factors[f1,:] += self.learning_rate*d

                        d_ranking = (-self.user_factors[u,:])*z
                        d = (1-self.alpha)*d_ranking - self.negative_feature_regularization*self.feature_user_factors[f2,:]
                        self.feature_user_factors[f2,:] += self.learning_rate*d

                        d_ranking = (self.item_factors[i,:])*z
                        d = (1-self.alpha)*d_ranking - self.positive_feature_regularization*self.feature_item_factors[f1,:]
                        self.feature_item_factors[f1,:] += self.learning_rate*d

                        d_ranking = (-self.item_factors[i,:])*z
                        d = (1-self.alpha)*d_ranking - self.negative_feature_regularization*self.feature_item_factors[f2,:]
                        self.feature_item_factors[f2,:] += self.learning_rate*d
            if abs(self.loss_hybrid() - old_loss) < 0.001 or self.loss_hybrid() - old_loss > 0:
                print 'iteration {0}: loss = {1}'.format(it, self.loss_hybrid())
                print 'converge!!'
                break
            else:
                old_loss = self.loss_hybrid()
                self.learning_rate *= 0.9
                print 'iteration {0}: loss = {1}'.format(it, self.loss_hybrid())

        #f = self.ValidateF1()
        f = self.ValidateF1_feature()
        return f
    def y_uif(self, uid, iid, fid):
        yuif = np.dot(self.user_factors[uid,:], self.feature_user_factors[fid,:])+np.dot(self.item_factors[iid,:], self.feature_item_factors[fid,:])
        return yuif
    def loss_hybrid(self):
        ranking_loss = 0
        rating_loss = 0
        complexity = 0

        for line in self.tensor_train.values:
            u = int(line[1].split('@')[0])
            i = int(line[1].split('@')[1])
            score = int(float(line[3]))
            rating_loss += (score - np.dot(self.user_factors[u],self.item_factors[i])) ** 2
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.item_regularization * np.dot(self.item_factors[i],self.item_factors[i])

            f1s = [f[0] for f in eval(line[2])]
            f2s = [i for i in range(self.num_features) if i not in f1s]
            '''
            for f1 in f1s:
                for f2 in f2s:
                    yuif_1 = self.y_uif(u,i,f1)
                    yuif_2 = self.y_uif(u,i,f2)
                    tmp = yuif_1 - yuif_2
                    ranking_loss += math.log(1.0+exp(-tmp))
                    complexity += self.positive_feature_regularization * np.dot(self.feature_user_factors[f1],self.feature_user_factors[f1])
                    complexity += self.positive_feature_regularization * np.dot(self.feature_item_factors[f1],self.feature_item_factors[f1])
                    complexity += self.negative_feature_regularization * np.dot(self.feature_user_factors[f2],self.feature_user_factors[f2])
                    complexity += self.negative_feature_regularization * np.dot(self.feature_item_factors[f2],self.feature_item_factors[f2])
            '''
        return self.alpha*rating_loss + (1-self.alpha)*ranking_loss + 0.5*complexity

    def gen_recommendation_feature(self, uid, iid, N):
        predict_scores = []
        for i in range(self.num_features):
            s = self.predict_feature(uid, iid, i)
            predict_scores.append([i, s])
        topN = np.argsort(np.array(predict_scores)[:,1])[-1:-N - 1:-1]
        return np.array(predict_scores)[topN][:, 0]
    def ValidateF1_feature(self):
        print 'evaluating...'
        recommendation_list = []
        purchased_list = []
        metric = BasicMetrics()
        for line in self.tensor_test.values:
            uid = int(line[1].split('@')[0])
            iid = int(line[1].split('@')[1])
            true_features = np.array(eval(line[2]))[:,0]
            #pre_features = self.user_favor.values[int(uid)][1:]
            pre_features = self.gen_recommendation_feature(uid, iid, 5)
            pre_features = [int(i) for i in pre_features]
            #self.print_user_pre_and_true_features(int(uid), int(iid), pre_features, true_features)
            recommendation_list.append(list(pre_features))
            purchased_list.append(list(true_features))
        p, r, f, h = metric.F1_score_Hit_ratio(recommendation_list, purchased_list)
        ndgg = metric.NDGG_k(recommendation_list, purchased_list)
        return f
    def print_user_pre_and_true_features(self, uid, iid, pre_features, true_features):
        item = self.tensor_items.values[iid][1]
        print 'item:' + str(item)
        print 'user predict features:'
        for i in pre_features:
            print self.tensor_features.values[i][1]
        print 'user true features'
        for j in true_features:
            print self.tensor_features.values[j][1]
    def predict_feature(self, uid, iid, fid):
        score = np.dot(self.user_factors[uid,:], self.feature_user_factors[fid,:])+np.dot(self.item_factors[iid,:], self.feature_item_factors[fid,:])
        return score

    def gen_recommendation(self, uid, N):
        predict_scores = []
        for i in range(self.num_items):
            s = self.predict(uid, i)
            predict_scores.append([i, s])
        topN = np.argsort(np.array(predict_scores)[:,1])[-1:-N - 1:-1]
        return np.array(predict_scores)[topN][:, 0]
    def ValidateF1(self):
        print 'evaluating...'
        recommendation_list = []
        purchased_list = []
        metric = BasicMetrics()

        user_all = list(set(np.array(self.ui_test)[:,0]))
        for uid in user_all:
            print uid
            user_items_index = np.argwhere(np.array(self.ui_test)[:,0] == int(uid))[:, 0]
            true_i = list(set(np.array(self.ui_test)[user_items_index][:, 1]))
            pre_i = self.gen_recommendation(uid, 5)
            pre_i = [(int(i)) for i in pre_i]
            print true_i, pre_i
            recommendation_list.append(list(true_i))
            purchased_list.append(list(pre_i))

        p, r, f, h = metric.F1_score_Hit_ratio(recommendation_list, purchased_list)
        ndgg = metric.NDGG_k(recommendation_list, purchased_list)
        t = pd.DataFrame([str(f), str(h), str(ndgg)])
        t.to_csv('bpr200_result')
        return f
    def print_user_favor_and_item_features(self, iid, features, feature_ratings):
        item = self.tensor_items.values[iid][1]
        print 'item:' + str(item)
        print 'user favorite features:'
        for i in features:
            print self.tensor_features.values[int(i[0])][1] + str(i[1])
        print 'item features'
        for j in feature_ratings:
            print self.tensor_features.values[j[0]][1] + str(j[1])
        raw_input()
    def predict(self, uid, iid):
        '''

        user_favor = []

        for fid in list(set(self.data.data[:,2])):
            fid_values = np.dot(self.user_factors[uid,:], self.feature_user_factors[fid,:])+np.dot(self.item_factors[iid,:], self.feature_item_factors[fid,:])
            user_favor.append([fid, fid_values])

        fN = 5
        index = np.argsort(np.array(user_favor)[:,1])[-1:-fN-1:-1]
        user_favor_feature = np.array(user_favor)[index]
        #nor = np.dot(user_favor_feature[:,1], user_favor_feature[:,1])
        #user_favor_feature = np.array([[i[0], i[1]/(nor**0.5)] for i in user_favor_feature])

        items = self.ratings.values[:, 1]
        index = np.argwhere(items == str(iid))[0][0]
        feature_ratings = eval(self.ratings.values[index][2])

        #self.print_user_favor_and_item_features(iid, user_favor_feature, feature_ratings)


        score = 0
        for f in user_favor_feature:
            if f[0] in np.array(feature_ratings)[:, 0]:
                #print 'match feature:' + str(self.tensor_features.values[int(f[0])][1])
                index = np.argwhere(np.array(feature_ratings)[:, 0] == f[0])[0][0]
                score += f[1]*np.array(feature_ratings)[index][1]

        a = 0.8
        s = a*np.dot(self.user_factors[uid,:], self.item_factors[iid,:]) + (1-a)*score
        '''
        return self.ave + self.item_bias[iid]+self.user_bias[uid]+np.dot(self.user_factors[uid,:], self.item_factors[iid,:])

if __name__ == '__main__':
    num_factors = 50
    f_tmp = 0
    for alpha in [0.01, 0.005]:
        for _ in range(100):
            print 'learning rate:' + str(alpha)
            print str(_)+' try...'
            model = TensorBPR(num_factors)
            num_iters = 5
            f = model.train(num_iters)
            raw_input()
            if f > f_tmp:
                print 'new_high!!!!'
                print f
                f_tmp = f
                t = pd.DataFrame([f, alpha])
                t.to_csv('learning_rate')
                t = pd.DataFrame(model.user_factors)
                t.to_csv('tensorbpr200pu')
                t = pd.DataFrame(model.item_factors)
                t.to_csv('tensorbpr200qi')
                t = pd.DataFrame(model.feature_user_factors)
                t.to_csv('tensorbpr200fu')
                t = pd.DataFrame(model.feature_item_factors)
                t.to_csv('tensorbpr200fi')
                t = pd.DataFrame(model.item_bias)
                t.to_csv('tensorbpr200b')




