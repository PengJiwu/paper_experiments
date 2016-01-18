__author__ = 'tao'

from sklearn.base import BaseEstimator
from sklearn import grid_search

from RRCP.MemeryDataModel import *
from Eval.Evaluation import *

'''
import sys
sys.path.insert(0, '../utils')
from IncompleteSVD import isvd
'''

from RRCP.utils import isvd
import Construction
import MyGaussian


class RRCP(BaseEstimator):
    def __init__(self, n=5, max_iter = 20, lambda_iter = 0.03, lambda_regular = 0.03):
        self.n = n
        self.max_iter = max_iter
        self.lambda_iter = lambda_iter
        self.lambda_regular = lambda_regular
        self.target_sample = []

    def model_score_without_log(self, alpha, beta, gamma, rate, price):
        return alpha * rate + beta * price**2 + gamma


    def sample_target(self):

        sample_ratio = 1
        for user_ix in xrange(self.dataModel.getUsersNum()):
            bought = list(self.dataModel.getData()[user_ix].nonzero()[1]) * sample_ratio
            origin_non_bought = [x for x in xrange(self.dataModel.getItemsNum()) if x not in bought]
            #print len(origin_non_bought), len(bought)
            non_bought_indices = random.sample(xrange(len(origin_non_bought)), len(bought))
            non_bought = [origin_non_bought[i] for i in non_bought_indices]
            samples = zip(bought, non_bought)
            self.target_sample.append(samples)
        '''
        for user_ix in xrange(self.dataModel.getUsersNum()):
            bought = list(self.dataModel.getData()[user_ix].nonzero()[1])
            origin_non_bought = [x for x in xrange(self.dataModel.getItemsNum()) if x not in bought]
            user_samples = []
            for bought_item in bought:
                for non_bought_item in origin_non_bought:
                    sample = (bought_item, non_bought_item)
                    user_samples.append(sample)
            self.target_sample.append(user_samples)
        '''

    def target_value(self, user_ix):
        result_value = 0.0
        if not self.target_sample:
            self.sample_target()
        samples = self.target_sample[user_ix]
        user_value = [(self.model_score(user_ix, i)-self.model_score(user_ix, j)) for i, j in samples]
        result_value += sum(user_value)

        regular = self.alpha[user_ix]**2 + self.beta[user_ix]**2
        result_value += regular

        return result_value

    def model_score(self, user_ix, item_ix):
        alpha = self.alpha[user_ix]
        beta = self.beta[user_ix]
        gamma = self.gamma[user_ix]
        rate = self.mf_rate[user_ix, item_ix]
        price = self.personal_item_price[user_ix][item_ix]

        prob = MyGaussian.get_prob_from_gaussian(self.dataModel.getUserByUid(user_ix), 'phones', self.personal_item_price[user_ix][item_ix], self.param_mu, self.param_sigm)
        result = alpha * rate + beta * np.exp(prob) + gamma
        return result


    def update(self, user_ix, item_i, item_j):
        ''' u * k
        C = U s V
        D =
        utility = rate_ui , price_ui
        utility  = alpha * rate - beta * price



        - utility_2

        rate_ui

        object = alpha_uc * (rate_1 - rate_2) + beta_uc * (price_1 - price_2)

        model_score_i = self.model_score_without_log(alpha, beta, gamma, rate_i, price_i)
        model_score_j = self.model_score_without_log(alpha, beta, gamma, rate_j, price_j)

        new_alpha = alpha - self.lambda_iter * ((rate_i * model_score_j - rate_j * model_score_i) / (model_score_j ** 2) - 2 * self.lambda_regular * alpha)
        new_beta = beta - self.lambda_iter * ((price_i**2 * model_score_j - price_j * model_score_i) / (model_score_j ** 2) - 2 * self.lambda_regular * beta)
        new_gamma = gamma - self.lambda_iter * ((model_score_j - model_score_i) / (model_score_j ** 2) - 2 * self.lambda_regular * gamma)
        '''
        alpha = self.alpha[user_ix]
        beta = self.beta[user_ix]
        rate_i = self.mf_rate[user_ix][item_i]
        rate_j = self.mf_rate[user_ix][item_j]
        prob_i = MyGaussian.get_prob_from_gaussian(self.dataModel.getUserByUid(user_ix), 'phones', self.personal_item_price[user_ix][item_i], self.param_mu, self.param_sigm)
        prob_j = MyGaussian.get_prob_from_gaussian(self.dataModel.getUserByUid(user_ix), 'phones', self.personal_item_price[user_ix][item_j], self.param_mu, self.param_sigm)

        new_alpha = alpha + self.lambda_iter * (rate_i - rate_j) - 2 * self.lambda_regular * alpha**2
        new_beta = beta + self.lambda_iter * (np.exp(prob_i) - np.exp(prob_j)) - 2 * self.lambda_regular * beta**2

        return new_alpha, new_beta

    def sample(self, user_ix, max_iter):
        item_num = self.dataModel.getItemsNum()
        bought = list(self.dataModel.getData()[user_ix].nonzero()[1])
        random.shuffle(bought)
        bought = bought[:max_iter]

        result = []
        while len(bought) + len(result) < item_num and len(result) < max_iter:
            for bought_item in bought:
                non_bought_item = bought_item
                while non_bought_item in bought or (bought_item, non_bought_item) in result:
                    non_bought_item = np.random.randint(item_num)
                result.append((bought_item, non_bought_item))
        random.shuffle(result)
        result = result[:int(math.ceil(len(result) * 0.7))]
        return result

    def fit(self, trainSamples, trainTargets):
        self.dataModel = MemeryDataModel(trainSamples, trainTargets)
        data = np.array(self.dataModel.getData().todense())
        u,s,v = isvd(data)
        new_data = np.dot(u, np.dot(s, v))
        new_data[new_data < 1] = np.nan
        u,s,v = isvd(new_data)
        new_data = np.dot(u, np.dot(s, v))
        #new_data[new_data < 1] = 0
        self.mf_rate = new_data

        train_data = pd.DataFrame(trainSamples, columns=['user', 'item', 'price'])
        item_price = dict(zip(train_data.ix[:,'item'], train_data.ix[:,'price']))
        data = train_data.groupby('user').mean()
        self.personal_item_price = np.empty((self.dataModel.getUsersNum(), self.dataModel.getItemsNum()))
        for user_ix in xrange(self.dataModel.getUsersNum()):
            user_avg_price = data.loc[self.dataModel.getUserByUid(user_ix), 'price']
            for item_ix in xrange(self.dataModel.getItemsNum()):
                delta_item_price = (item_price[self.dataModel.getItemByIid(item_ix)] - user_avg_price) / user_avg_price
                self.personal_item_price[user_ix][item_ix] = item_price[self.dataModel.getItemByIid(item_ix)]

        b = {'phones':train_data}
        price_df = Construction.get_user_category_buy_price(b)
        self.param_mu, self.param_sigm = MyGaussian.gaussian_curve_fit(price_df)

        self.alpha = np.random.rand(self.dataModel.getUsersNum())
        self.beta = np.random.rand(self.dataModel.getUsersNum())
        self.gamma = np.zeros(self.dataModel.getUsersNum())

        origin_lambda_iter = self.lambda_iter
        for user_ix in xrange(self.dataModel.getUsersNum()):
            samples = self.sample(user_ix, self.max_iter)
            self.lambda_iter = origin_lambda_iter
            old_target_value = 0
            for item_1, item_2 in samples:
                new_alpha, new_beta = self.update(user_ix, item_1, item_2)

                ##if not old_target_value or old_target_value < target_value:
                    #old_target_value = target_value
                self.alpha[user_ix] = new_alpha
                self.beta[user_ix] = new_beta
                self.lambda_iter = self.lambda_iter * 0.9
                #else:
                    #break
                #print user_ix, target_value

                #self.gamma[user_ix] = new_gamma
                #print user_ix, self.target_value(user_ix)

            #print user_ix, self.target_value(user_ix)
        self.lambda_iter = origin_lambda_iter
        self.beta = np.zeros(self.dataModel.getUsersNum())

    def recommend(self, u):
        uid = self.dataModel.getUidByUser(u)
        predict_scores = []
        for i in range(self.dataModel.getItemsNum()):
            predict_scores.append(self.model_score(uid, i))
        topN = np.argsort(np.array(predict_scores))[-1:-self.n-1:-1]
        for item in topN:
            price = self.personal_item_price[uid][item]
            #print self.mf_rate[u, item], MyGaussian.get_prob_from_gaussian(uid, 'phones', price, self.param_mu, self.param_sigm)
        return [self.dataModel.getItemByIid(i) for i in topN]

    def score(self, testSamples, trueLabels=None):
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            '''
            if not self.dataModel.getUidByUser(u) == -1:
                for item in uTrueIndex:
                    price = testSamples[item][2]
                    print self.mf_rate[self.dataModel.getUidByUser(u), self.dataModel.getIidByItem(testSamples[item][1])], MyGaussian.get_prob_from_gaussian(u, 'phones', price, self.param_mu, self.param_sigm)
            '''
            if not self.dataModel.getUidByUser(u) == -1:
                trueList.append(true)
                pre = self.recommend(u)
                recommendList.append(pre)
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'RRCP result:'+'('+str(self.get_params())+'):\t' + str((result)['F1'])
        return (result)['F1']

if __name__ == '__main__':
    rrcp = RRCP()
    for post in reversed(['_filter_5_format.csv']):
        filename = '/Users/tao/Documents/coding/dataset/AmazonMeta/reviews_cell_phones' + post
        #filename = '../Data/review/reviews_cell_phones' + post
        print filename

        train_data = pd.read_csv(filename)
        trainsamples = [[int(i[0]), int(i[1]), int(i[2])] for i in train_data.values]
        targets = [x[4] for x in train_data.values]

        parameters = {'n':[5], 'max_iter':[5, 15], 'lambda_iter':[0.7, 0.03], 'lambda_regular':[0.03]}

        clf = grid_search.GridSearchCV(rrcp, parameters,cv=3)
        clf.fit(trainsamples, targets)
        print clf.best_score_, clf.best_params_

        '''
        rrcp.fit(trainsamples, targets)

        print rrcp.alpha, rrcp.beta

        test_data = pd.read_csv('/Users/tao/Documents/coding/dataset/AmazonMeta/reviews_cell_phones_filter_3_test.csv')
        testsamples = [[int(i[0]), int(i[1])] for i in test_data.values]

        score = rrcp.score(testsamples)
        print score
        '''


