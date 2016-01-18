#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import pandas as pd
import numpy as np
import ast
import sys
from collections import Counter


def filter_low_categories_users(dfs, lowest_user_categories, lowest_category_reviews):
    real_users = [df['user'].tolist() for df in dfs]

    users = set([])
    for category_users in real_users:
        users = users.union(category_users)

    real_users = [Counter(x) for x in real_users]

    print 'users', len(users)
    valid_users = []
    for user in users:
        num = 0
        for category_users in real_users:
            if category_users[user] >= lowest_category_reviews:
                num += 1
            if num >= lowest_user_categories:
                valid_users.append(user)
                break

    print 'valid users', len(valid_users)

    result = []
    for df in dfs:
        result.append(df[df.user.isin(valid_users)])
    return result

def filter_low_reviews_items(dfs, lowest_item_reviews):
    df = pd.concat(dfs)
    df = df.groupby('item').agg('count')
    df = df[df.rate >= lowest_item_reviews]
    valid_items = set(df.index)
    print 'valida items', len(valid_items)

    result = []
    for df in dfs:
        result.append(df[df.item.isin(valid_items)])
    return result


def read_data(input_dir, types, input_meta, input_profile):
    dfs = []
    for type in types:
        type_meta_path = input_dir + type + input_meta
        type_profile_path = input_dir + type + input_profile

        meta = pd.read_csv(type_meta_path)
        size = meta.shape[0]
        meta.index = range(size)
        with open(type_profile_path, 'r') as f:
            lines = f.readlines()

        df_data = []
        size = len(lines)
        for i in range(size):
            if not lines[i].strip():
                continue
            sub_data = [meta.ix[i, 'user'], meta.ix[i, 'item'], lines[i].strip(), meta.ix[i, 'rate']]
            df_data.append(sub_data)

        df = pd.DataFrame(df_data, columns=['user', 'item', 'feature', 'rate'])
        df['rate'] = df['rate'].astype(int)
        dfs.append(df)

        print 'df', df.shape

    return dfs



def write_data(dfs, output_dir, types, output_result):
    for i in range(len(dfs)):
        type_result_path = output_dir + types[i] + output_result
        dfs[i].to_csv(type_result_path, index=False)

    df = pd.concat(dfs)
    df = df.iloc[np.random.permutation((len(df)))]
    df.reset_index(drop=True)
    df.to_csv(output_dir + 'all.csv', index=False)


def main():
    #input_dir = './YelpReviews/'
    input_dir = './AmazonReviews/'
    output_dir = input_dir
    #types = ['beauty', 'entertainment', 'food', 'health', 'yelp_clothing', 'bars']
    types = ['amazon_clothing', 'baby', 'office', 'phones', 'toys']
    input_meta = '_meta.csv'
    input_profile = '.profile'
    output_result = '_result.csv'

    lowest_user_categories = 3
    lowest_category_reviews = 3
    lowest_item_reviews = 5

    dfs = read_data(input_dir, types, input_meta, input_profile)
    dfs = filter_low_categories_users(dfs, lowest_user_categories, lowest_category_reviews)
    dfs = filter_low_reviews_items(dfs, lowest_item_reviews)

    reload(sys)
    sys.setdefaultencoding('utf-8')
    write_data(dfs, output_dir, types, output_result)

if __name__ == '__main__':
    main()
