#-*- coding:utf-8 -*-

__author__ = 'tao'

import math

import pandas as pd
import numpy as np


def _read_train_or_test(path):
    file = pd.read_csv(path)

    '''
    del file['Unnamed: 0']
    file.columns = 'user', 'item', 'str_feature', 'rate'
    feature_series = file['str_feature'].map(ast.literal_eval)
    file['feature'] = feature_series
    del file['str_feature']
    '''
    feature_series = file['feature'].map(eval)
    file['feature'] = feature_series
    return file

def _get_user_or_item_or_feature_num(path):
    file = pd.read_csv(path)
    return file.shape[0]

def _construct_A_apply(A, x):
    A.ix[x['user'], x['item']] = x['rate']

def _construct_A(train_data, user_num, item_num):
    A = pd.DataFrame(np.zeros((user_num, item_num)))

    train_data.apply((lambda x: _construct_A_apply(A, x)), axis=1)

    return A

def _construct_X_apply(user_feature_mention, x):
    for feature in x['feature']:
        user_feature_mention.ix[x['user'], feature[0]] += 1


def _construct_X(train_data, user_num, feature_num):
    X = pd.DataFrame(np.zeros((user_num, feature_num)))
    user_feature_mention = pd.DataFrame(np.zeros((user_num, feature_num)))

    train_data.apply((lambda x : _construct_X_apply(user_feature_mention, x)), axis=1)
    for idx, row in user_feature_mention.iterrows():
        for rdx in range(feature_num):
            if not row[rdx] == 0:
                X.ix[idx, rdx] = 1 + (5 - 1) * (2 / (1 + math.exp(-row[rdx])) - 1)

    return X

def _construct_Y_apply(item_feature_mention_times, item_feature_mention_avg_sentiment, x):
    for feature in x['feature']:
        sentiment_sum = item_feature_mention_avg_sentiment.ix[x['item'], feature[0]] * item_feature_mention_times.ix[x['item'], feature[0]]
        sentiment_sum += feature[1]
        item_feature_mention_times.ix[x['item'], feature[0]] += 1
        item_feature_mention_avg_sentiment.ix[x['item'], feature[0]] = sentiment_sum / item_feature_mention_times.ix[x['item'], feature[0]]


def _construct_Y(train_data, item_num, feature_num):
    Y = pd.DataFrame(np.zeros((item_num, feature_num)))
    item_feature_mention_times = pd.DataFrame(np.zeros((item_num, feature_num)))
    item_feature_mention_avg_sentiment = pd.DataFrame(np.zeros((item_num, feature_num)))

    train_data.apply((lambda x : _construct_Y_apply(item_feature_mention_times, item_feature_mention_avg_sentiment, x)), axis=1)
    for idx, row in item_feature_mention_times.iterrows():
        for rdx in range(feature_num):
            if not row[rdx] == 0:
                Y.ix[idx, rdx] = 1 + (5 - 1) / (1 + math.exp(-(item_feature_mention_times.ix[idx, rdx] * item_feature_mention_avg_sentiment.ix[idx, rdx])))

    '''
    item_feature_mention_time_sum = 0
    for idx, row in item_feature_mention_times.iteritems():
        for val in row:
            if not val == 0:
                item_feature_mention_time_sum += 1
    item_feature_mention_avg_times = item_feature_mention_time_sum / item_feature_mention_times.shape[0]
    column_index = get_top_columns_index(Y, item_feature_mention_avg_times)
    for idx, row in Y.iterrows():
        for rdx, val in row.iteritems():
            if not rdx in column_index[idx]:
                row[rdx] = 0
    '''
    return Y

def _construct_buy_set(test_data):
    result = set(zip(test_data['user'], test_data['item']))
    return result

def _construct_mention_feature_apply(result, row):
    ui = (row['user'], row['item'])
    feature = [item[0] for item in row['feature']]

    if not ui in result:
        result[ui] = set([])
    result[ui] = result[ui].union(feature)

def _construct_mention_feature(test_data):
    result = {}

    test_data.apply((lambda x : _construct_mention_feature_apply(result, x)), axis=1)
    return result

def construction(path):
    #path = '~/Downloads/v3/'
    #path = '/Users/tao/Documents/coding/dataset/YelpReviews/'
    #path = './YelpReviews/'

    train_data = _read_train_or_test(path + 'train.csv')
    test_data = _read_train_or_test(path + 'test.csv')

    user_num = _get_user_or_item_or_feature_num(path + 'user.csv')
    item_num = _get_user_or_item_or_feature_num(path + 'item.csv')
    feature_num = _get_user_or_item_or_feature_num(path + 'feature.csv')

    paper_symbol_A = _construct_A(train_data, user_num, item_num)
    paper_symbol_X = _construct_X(train_data, user_num, feature_num)
    paper_symbol_Y = _construct_Y(train_data, item_num, feature_num)

    train_buy_set = _construct_buy_set(train_data)
    real_result = _construct_buy_set(test_data)
    test_mention_feature = _construct_mention_feature(test_data)

    return paper_symbol_A, paper_symbol_X, paper_symbol_Y, user_num, item_num, feature_num, train_buy_set,  real_result, test_mention_feature
