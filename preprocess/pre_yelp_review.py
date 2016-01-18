#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import json
import numpy as np
import pandas as pd
import sys
import operator

def main():
    lowest_user_reviews = 2
    lowest_product_reviews = 2
    input_dir = '/Users/tao/Documents/coding/dataset/yelp_dataset_challenge_academic_dataset/'
    input_business = input_dir + 'yelp_academic_dataset_business.json'
    input_review = input_dir + 'yelp_academic_dataset_review.json'
    output_dir = input_dir + 'output/'
    output_meta = '.meta.pkl'
    output_review = '.review'
    output_product = '.product'

    category_list = ['Entertainment', 'Food', 'Clothing', 'Health_Medical', 'Beauty', 'Bars']
    category_symbol = [ ["Arts & Entertainment"],
                        ["Food", "Fast Food", "Pizza", "Burgers", "Sandwiches"],
                        ["Men's Clothing", "Women's Clothing", "Children's Clothing", "Jewelry", "Fashion"],
                        ["Doctors", "Health & Medical"],
                        ["Beauty & Spas"],
                        ["Bars", "Nightlife"]
    ]
    category_size = len(category_list)

    reload(sys)
    sys.setdefaultencoding('utf-8')

    business_json = []
    review_json = []
    with open(input_business, 'r') as f:
        for line in f.readlines():
                try:
                    json_data = json.loads(line)
                except ValueError:
                    continue
                business_json.append(json_data)
    with open(input_review, 'r') as f:
        for line in f.readlines():
                try:
                    json_data = json.loads(line)
                except ValueError:
                    continue
                review_json.append(json_data)
    '''
    i = 0
    for item in business_json:
        if "Home Services" in item['categories']:
            i += 1
            print item['categories'], i

    a = list(set([y for x in business_json for y in x['categories']]))
    tmp = dict(zip(a, [0]*len(a)))
    for item in business_json:
        for c in item['categories']:
            tmp[c] += 1
    tmp = sorted(tmp.items(), key=operator.itemgetter(1))
    for i in tmp:
        print i
    '''


    selected_business_id = [[] for _ in range(category_size)] #add element to list of list independently
    for item in business_json:
        ever_add_ix = -2
        for ix in range(category_size):
            for cs in category_symbol[ix]:
                categories = item['categories']
                if cs in categories and ever_add_ix == -2:
                    selected_business_id[ix].append(item['business_id'])
                    ever_add_ix = ix
                    break
                elif cs in categories and not ever_add_ix == -1:
                    #如果曾经加过，说明该item属于多类，抛弃之
                    selected_business_id[ever_add_ix].remove(item['business_id'])
                    ever_add_ix = -1

    for ix in range(category_size):
        category = category_list[ix]
        single_category_business_id = selected_business_id[ix]

        review = [x for x in review_json if x['business_id'] in single_category_business_id]
        df_data = [[x['user_id'], x['business_id'], x['stars'], x['text']] for x in review]
        df_data = np.array(df_data)
        if df_data.shape[0] < lowest_user_reviews:
            continue
        columns = ['user', 'item', 'rate', 'review']
        df = pd.DataFrame(df_data, columns=columns)

        #Filtering
        df = df[df.review != '']
        filter_df = df.groupby('user').agg('count')
        wanted = filter_df[filter_df['rate'] >= lowest_user_reviews].index
        df = df[df.user.isin(wanted)]
        filter_df = df.groupby('item').agg('count')
        wanted = filter_df[filter_df['rate'] >= lowest_product_reviews].index
        df = df[df.item.isin(wanted)]
        if df.shape[0] < lowest_user_reviews:
            continue
        df.index = range(df.shape[0])
        filter_df = 0
        wanted = 0

        print category, df.shape

        df_group_list = []
        f0 = open(output_dir + category + output_review, 'w')
        with open(output_dir + category + output_product, 'w') as f1:
            for item_value, item_df in df.groupby('item'):
                df_group_list.append(item_df)
                f1.write(item_value + '\t' + str(item_df.shape[0]) + '\t' + 'url\n')
                for review in item_df['review']:
                    f0.write('<DOC>\n')
                    f0.write(review)
                    f0.write('\n</DOC>\n')
                    f1.write('\t<DOC>\n\t')
                    f1.write(review)
                    f1.write('\n\t</DOC>\n')
        f0.close()
        df = 0

        df_group = pd.concat(df_group_list)
        df_group.to_pickle(output_dir + category + output_meta)
        df_group = 0


if __name__ == '__main__':
    main()
