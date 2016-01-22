#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import pandas as pd
import numpy as np
import os.path
import math
from random import shuffle


def generate_uif(df, user_file, item_file):
    users = df['user']
    items = df['item']

    t = pd.DataFrame(list(set(users)), columns=['user'])
    t.to_csv(user_file, index=False)
    t = pd.DataFrame(list(set(items)), columns=['item'])
    t.to_csv(item_file, index=False)


def format_with_id(df, format_file, user_file, item_file):
    if os.path.isfile(format_file):
        new_df = pd.read_csv(format_file)
        return new_df
    user_data = pd.read_csv(user_file)
    users = np.array(user_data.values[:,0])
    item_data = pd.read_csv(item_file)
    items = np.array(item_data.values[:,0])
    result = []
    ix = -1


    for line in df.values:
        ix += 1
        if ix % 1000 == 0:
            print ix
        uid = np.argwhere(users == line[0])[0][0]
        iid = np.argwhere(items == line[1])[0][0]

        result.append([uid, iid, line[2], line[3], line[4], line[5]])

    shuffle(result)
    t = pd.DataFrame(result, columns=['user', 'item', 'title', 'price', 'rate', 'time'])
    t.to_csv(format_file, index=False)
    return t

def gen_train_and_test(df, train_file, test_file):
    df = df.sort(['time'])

    users_all = np.array(df.values)[:,0]
    users_set = list(set(users_all))
    users_records_num = [list(users_all).count(i) for i in range(len(users_set))]
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
    for line in lines:
        uid = int(line[0])
        if user_num[uid] < train_num[uid]:
            user_num[uid] += 1
            train.append(line)
        elif line[1] in valid_items:
            test.append(line)

    #过滤不在train的物品里的test case
    #valid_item = set([line[1] for line in train])
    #valid_test = [line for line in test if line[1] in valid_item]

    t = pd.DataFrame(train, columns=['user', 'item', 'title', 'price', 'rate', 'time'])
    t.to_csv(train_file, index=False)
    t = pd.DataFrame(test, columns=['user', 'item', 'title', 'price', 'rate', 'time'])
    t.to_csv(test_file, index=False)

if __name__ == '__main__':
    filename = '/Users/tao/Documents/coding/dataset/workplace/phonesu5i5'
    df = pd.read_csv(filename + '.csv')
    train_file = filename + '_train.csv'
    test_file = filename + '_test.csv'
    user_file = filename + '_user.csv'
    item_file = filename + '_item.csv'
    format_file = filename + '_format.csv'

    generate_uif(df, user_file, item_file)
    df = format_with_id(df, format_file, user_file, item_file)
    gen_train_and_test(df, train_file, test_file)

    #df = pd.read_csv(train4test_file)
    #gen_train_and_test(df, train_file, validate_file)
    '''
    #input_dir = '/Users/tao/Documents/coding/dataset/YelpReviews/product_review5_category3_item4/'
    #types = ['bars', 'beauty', 'entertainment', 'health', 'yelp_clothing', 'food']
    input_dir = '/Users/tao/Documents/coding/dataset/AmazonReviews/product_review15_category3_item4/'
    types = ['amazon_clothing', 'baby', 'office', 'phones', 'toys']
    categorys = []
    for type in types:
        file = input_dir + type + '_result.csv'
        df = pd.read_csv(file)
        categorys.append(set(df['item']))
    #input_dir = './YelpReviews/'
    for ix in range(1,6):
        print 'Now is ix of ' + str(ix)
        output_dir = input_dir + str(ix) + '/'
        source_file = output_dir + 'all.csv'
        user_file = output_dir + 'user.csv'
        item_file = output_dir + 'item.csv'
        feature_file = output_dir + 'feature.csv'
        format_file = output_dir + 'format.csv'
        train_file = output_dir + 'train.csv'
        test_file = output_dir + 'test.csv'

        df = pd.read_csv(source_file)
        generate_uif(df, user_file, item_file, feature_file)
        df = format_with_id(df, format_file, user_file, item_file, feature_file, categorys)
        gen_train_and_test(df, train_file, test_file)
    '''