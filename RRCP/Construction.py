__author__ = 'tao'
import pandas as pd
import numpy as np
import math

import MyGaussian


def get_user_category_buy_price(df, dataModel):
    df_data = [[] for _ in xrange(dataModel.getUsersNum())]
    df_group = df.groupby('user')
    for ix in xrange(dataModel.getUsersNum()):
            user_df = df_group.get_group(dataModel.getUserByUid(ix))
            price = []
            for idx, row in user_df.iterrows():
                price.append(row['price'])
            df_data[ix] = price
    return df_data

def _construct_paramenter():
    pass

def _construct_mention_feature_apply(dataModel, origin_user_price_feature, row, user_sigm):
    price_ix = dataModel.getPidByPriceIx(row['price_ix'])
    feature = [dataModel.getFidByFeature(i[0]) for i in eval(row['feature'])]

    min_ix = int(price_ix - user_sigm)
    max_ix = int(price_ix + user_sigm)
    if min_ix < 0:
        min_ix = 0
    if max_ix > dataModel.getPriceIxNum():
        max_ix = dataModel.getPriceIxNum()

    for i in xrange(min_ix, max_ix):
        origin_user_price_feature[i][feature] = origin_user_price_feature[i][feature] + MyGaussian.get_prob_from_gaussian(i, price_ix, user_sigm)

def construct_price_feature(df, dataModel):
    #category_df = {'phones':df}
    price_df = get_user_category_buy_price(df, dataModel)
    param_mu, param_sigm = MyGaussian.gaussian_curve_fit(price_df)
    result = {}
    for user, user_df in df.groupby('user'):
        user_sigm = np.sqrt(param_sigm[dataModel.getUidByUser(user)])
        origin_user_price_feature = np.zeros((dataModel.getPriceIxNum(), dataModel.getFeaturesNum()))
        user_df.apply((lambda x : _construct_mention_feature_apply(dataModel, origin_user_price_feature, x, user_sigm)), axis=1)
        result[dataModel.getUidByUser(user)] = origin_user_price_feature
    return result

def construct_item_feature(df, dataModel):
    item_num = dataModel.getItemsNum()
    feature_num = dataModel.getFeaturesNum()
    result = np.zeros((item_num, feature_num))
    item_feature_sentiment = [[[] for _ in range(feature_num)] for _ in range(item_num)]
    item_feature_mention_time = np.zeros(item_num)
    for ix, row in df.iterrows():
        item = dataModel.getIidByItem(row['item'])
        features = eval(row['feature'])
        for feature in features:
            item_feature_sentiment[item][dataModel.getFidByFeature(feature[0])].append(feature[1])
            item_feature_mention_time[item] += 1

    rating = np.array(dataModel.getData())
    for ix in xrange(item_num):
        review_num = np.count_nonzero(rating[:,ix])
        for rx in xrange(feature_num):
            #average sentiment every reviewer
            avg_sentiment = sum(item_feature_sentiment[ix][rx])/(review_num + 0.0)
            result[ix,rx] = 1/(1+np.exp(-avg_sentiment))
    return result