__author__ = 'tao'

from sklearn.base import BaseEstimator
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold
import numpy as np

import MyMemeryDataModel
from Eval.Evaluation import *
import TensorFactorization
import Construction


class RRCP(BaseEstimator):
    def __init__(self, n=5, max_iter=5, implict_dim_num=5, pf_lambda=0.5, train_sentry=1, learning_rate=0.1, regularization=0.03):
        self.n = n
        self.max_iter = max_iter
        self.implict_dim_num = implict_dim_num
        self.pf_lambda = pf_lambda
        self.train_sentry = train_sentry
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, trainSamples, trainTargets):
        self.trainDataFrame = pd.DataFrame(trainSamples, columns = ['user', 'item', 'price_ix', 'price', 'feature', 'rate'])
        self.item_price_ix_dict = self.trainDataFrame.set_index('item')['price_ix'].to_dict()
        self.dataModel = MyMemeryDataModel.MemeryDataModel(trainSamples, trainTargets)

        tensor_model = TensorFactorization.TensorBPR(self.trainDataFrame, self.dataModel, self.implict_dim_num, self.max_iter, self.pf_lambda, self.train_sentry, self.learning_rate, self.regularization)
        user_price_feature_tensor = tensor_model.train()

        '''
        for price_feature in user_price_feature_tensor:
            for features in price_feature:
                feature_ix = sorted(range(self.dataModel.getFeaturesNum()), key=lambda x: features[x], reverse=True)[self.most_like_feature_num:]
                features[feature_ix] = 0
        '''
        item_feature = Construction.construct_item_feature(self.trainDataFrame, self.dataModel)

        self.user_price_item_tensor = np.dot(user_price_feature_tensor, item_feature.transpose())

        return self.user_price_item_tensor

    def recommend(self, u):
        price_item = self.user_price_item_tensor[self.dataModel.getUidByUser(u)]
        utility = {}
        for item, price_ix in self.item_price_ix_dict.iteritems():
            utility[item] = price_item[self.dataModel.getPidByPriceIx(price_ix), self.dataModel.getIidByItem(item)]
        recommend_items = [x[0] for x in sorted(utility.items(), key=lambda (k,v): v)][-1:-self.n-1:-1]
        return recommend_items

    def score(self, testSamples, trueLabels=None):
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            pre = self.recommend(u)
            #print pre, true
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'RRCP result:'+'('+str(self.get_params())+'):\t' + str((result)['F1'])
        return (result)['F1']

if __name__ == '__main__':
    rrcp = RRCP()
    filepath = '~/Documents/coding/dataset/workplace/filter_phones_format.csv'
    #filepath = 'filter_phones_format.csv'
    data = pd.read_csv(filepath)
    samples = data.values
    targets = [int(i[3]) for i in samples]
    #parameters = {'n':[5], 'max_iter':[200], 'implict_dim_num':[50, 100], 'train_sentry':[200, 100, 50, 10, 1], 'pf_lambda':[0.99, 0.9, 0.5, 0.1, 0.01], 'learning_rate':[0.3, 0.1, 0.03, 0.01], 'regularization':[0.1, 0.03, 0.01]}

    parameters = {'n':[5], 'max_iter':[200], 'implict_dim_num':[50, 100], 'train_sentry':[100, 10, 1], 'pf_lambda':[10, 1, 0.1], 'learning_rate':[0.3, 0.1, 0.03, 0.01], 'regularization':[0.1, 0.03, 0.01]}

    labels = [i[0] for i in data.values]
    rec_cv = StratifiedKFold(labels, 2)
    clf = grid_search.GridSearchCV(rrcp, parameters,cv=rec_cv)
    clf.fit(samples, targets)
    print(clf.grid_scores_)
    print clf.best_score_, clf.best_params_


