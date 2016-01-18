#-*- coding:utf-8 -*-

__author__ = 'tao'

import math

import pandas as pd
from collections import Counter


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

def _construct_buy_set(test_data):
    result = set(zip(test_data['user'], test_data['item']))
    return result

def _construct_mention_feature_list_apply(row, result):
    for item in row['feature']:
        result.append(item[0])

def _construct_mention_feature_list(test_data):
    result = []
    test_data.apply((lambda row: _construct_mention_feature_list_apply(row, result)), axis=1)

    return result

def _construct_mention_buy_list(test_data):
    result = list(test_data['item'])
    return result

def _most_common_feature_dict(user_data, feature_list, feature_k):
    result = {}
    ct = Counter(feature_list)
    rl = ct.most_common(feature_k)
    features = set([item[0] for item in rl])
    for user in user_data.index:
        result[user] = features

    return result

def _most_common_buy_set(user_data, feature_list, feature_k):

    result = set([])
    ct = Counter(feature_list)
    rl = ct.most_common(feature_k)
    for item in rl:
        for user in user_data.index:
            result.add((user, item[0]))

    return result

def _construct_mention_feature_dict_apply(result, row):
    ui = (row['user'], row['item'])
    feature = [item[0] for item in row['feature']]

    if not ui in result:
        result[ui] = set([])
    result[ui] = result[ui].union(feature)

def _construct_mention_feature_dict(test_data):
    result = {}

    test_data.apply((lambda x : _construct_mention_feature_dict_apply(result, x)), axis=1)
    return result

def _construct_buy_set(test_data):
    result = set(zip(test_data['user'], test_data['item']))
    return result

def construction(path):
    user_data = pd.read_csv(path + 'user.csv')
    train_data = _read_train_or_test(path + 'train.csv')
    test_data = _read_train_or_test(path + 'test.csv')

    train_metion_feature = _construct_mention_feature_list(train_data)
    top_N_feature_dict = _most_common_feature_dict(user_data, train_metion_feature, 10)
    test_mention_feature_dict = _construct_mention_feature_dict(test_data)

    train_buy_list = _construct_mention_buy_list(train_data)
    top_N_buy_set = _most_common_buy_set(user_data, train_buy_list, 10)
    test_buy_set = _construct_buy_set(test_data)

    return top_N_feature_dict, test_mention_feature_dict