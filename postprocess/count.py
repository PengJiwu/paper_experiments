#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import pandas as pd
from itertools import combinations
import ast
from collections import Counter

def _load_data(types, input_dir):
    #types = ['home', 'baby', 'music', 'office', 'phones', 'toys']
    #types = ['baby']
    postfix = '_result.csv'

    result = []
    for type in types:
        file = input_dir + type + postfix
        data = pd.read_csv(file)
        result.append(data)

    return result

def _user_num(dfs):
    nums = set([])
    l = []
    for df in dfs:
        num = df.groupby('user').agg('count').index
        l.append(len(num))
        nums = nums.union(num)

    return len(nums), l

def _product_num(dfs):
    nums = set([])
    l = []
    for df in dfs:
        num = df.groupby('item').agg('count').index
        l.append(len(num))
        nums = nums.union(num)

    return len(nums), l


def _review_num(dfs):
    nums = []
    for df in dfs:
        num = df.shape[0]
        nums.append(num)

    return sum(nums), nums

def _several_popular_feature(dfs, num):
    result = []
    for df in dfs:
        feature = df['feature'].tolist()
        feature = [ast.literal_eval(x) for x in feature]
        feature = [y[0] for x in feature for y in x]
        data = Counter(feature)
        common = data.most_common(num)
        feature = [x for x in common]
        result.append(feature)

    return result

def _all_popular_feature(dfs, num):
    df = pd.concat(dfs)
    feature = df['feature'].tolist()
    feature = [ast.literal_eval(x) for x in feature]
    feature = [y[0] for x in feature for y in x]
    data = Counter(feature)
    common = data.most_common(num)
    feature = [x for x in common]

    return feature

def _categories_users(dfs, nums):
    categories_users = [set(df['user']) for df in dfs]

    all_users = set([])
    for category_users in categories_users:
        all_users = all_users.union(category_users)

    result = []
    for num in nums:
        user_num = 0
        for user in all_users:
            category_num = 0
            for category_users in categories_users:
                if user in category_users:
                    category_num += 1
                if category_num >= num:
                    user_num += 1
                    break

        result.append(user_num)
    return result


if __name__ == '__main__':
    input_dir = '/Users/tao/Documents/coding/dataset/AmazonReviews/product_review15_category3_item4/'
    #input_dir = '/Users/tao/Documents/coding/dataset/YelpReviews/product_review5_category3_item4/'
    types = ['phones', 'office', 'toys', 'amazon_clothing', 'baby']
    #types = ['bars', 'beauty', 'entertainment', 'food', 'health', 'yelp_clothing']
    dfs = _load_data(types, input_dir)


    all_pf = _all_popular_feature(dfs, 10)
    print "all count", all_pf
    popular_feautre = _several_popular_feature(dfs, 10)
    for ix in range(len(types)):
        print types[ix], popular_feautre[ix]


    #用户物品评论计数
    user_num = _user_num(dfs)
    product_num = _product_num(dfs)
    review_num = _review_num(dfs)
    print 'user number', user_num
    print 'product number', product_num
    print 'review number', review_num

    #交叉用户数
    nums = [1, 2, 3, 4, 5, 6]
    inter_nums = _categories_users(dfs, nums)
    for ix in range(len(nums)):
        print nums[ix], inter_nums[ix]