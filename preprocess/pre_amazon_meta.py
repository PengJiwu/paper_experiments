#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import numpy as np
import pandas as pd
import json

def main():
    #input_dir = '/Users/tao/Documents/coding/dataset/workplace/'
    input_dir = '/home/qzlab/tao_workstation/price/'
    output_dir = input_dir
    piece_len = 500000

    meta_categories = ['Clothing_Shoes_and_Jewelry2']
    review_categories = ['Clothing_Shoes_and_Jewelry2']
    input_category_meta_name = []
    output_category_meta_item_name = []
    output_category_meta_user_name = []
    input_category_review_name = []
    output_category_review_name = []
    output_category_name = []
    for ix in range(len(meta_categories)):
        input_category_meta_name.append(input_dir + 'meta_' + meta_categories[ix] + '.json')
        #output_category_meta_item_name.append(output_dir + 'meta_' + meta_categories[ix] + '.csv')
        #output_category_meta_user_name.append(output_dir + 'meta_' + meta_categories[ix] + '_user.csv')

        input_category_review_name.append(input_dir + 'reviews_' + review_categories[ix] + '.json')
        #output_category_review_name.append(input_dir + 'reviews_' + meta_categories[ix] + '.csv')
        output_category_name.append(input_dir + meta_categories[ix] + '.csv')


    for ix in range(len(meta_categories)):
        with open(input_category_meta_name[ix], 'r') as f:
            valid_items = []
            data = f.readlines()
            for item in data:
                item_dict = eval(item)
                if 'price' in item_dict and 'title' in item_dict:
                    valid_items.append(item_dict)
            '''
            valid_items = json.load(f)
            '''

        df_data = np.array([[x['asin'], x['title'], float(x['price'])] for x in valid_items])
        column_name = ['item', 'title', 'price']
        df = pd.DataFrame(df_data, columns=column_name)
        valid_item_asin = set(df['item'])
        #Fileter
        #df = df[df.price >= 5]
        #df = df[df.price < 300]
        #df.to_csv(output_category_meta_item_name[ix], index=False, encoding='utf-8')

        item_price = df.set_index('item')['price'].to_dict()
        item_title = df.set_index('item')['title'].to_dict()
        index_item_dict = df['item'].to_dict()
        item_index_dict = {v:k for k, v in index_item_dict.items()}
        index_item_dict = None

        dfs = []
        with open(input_category_review_name[ix], 'r') as f:
            data = f.readlines()
            data = [eval(x) for x in data]
            pieces = [data[x:x+piece_len] for x in xrange(0, len(data), piece_len)]
            for piece_json in pieces:
                '''
                print 0
                piece_json = []
                for line in piece:
                    try:
                        json_data = eval(line)
                    except ValueError:
                        continue
                    piece_json.append(json_data)
                '''

                df_data1 = np.array([[x['reviewerID']] for x in piece_json])
                df_data2 = np.array([[x['asin']] for x in piece_json])
                df_data3 = np.array([[x['asin']] for x in piece_json])
                df_data4 = np.array([[x['asin']] for x in piece_json])
                df_data5 = np.array([[x['overall']] for x in piece_json])
                df_data6 = np.array([[x['unixReviewTime']] for x in piece_json])
                columns1 = ['user']
                df1 = pd.DataFrame(df_data1, columns=columns1)
                columns2 = ['item']
                df2 = pd.DataFrame(df_data2, columns=columns2)
                columns3 = ['title']
                df3 = pd.DataFrame(df_data3, columns=columns3)
                columns4 = ['price']
                df4 = pd.DataFrame(df_data4, columns=columns4)
                columns5 = ['rate']
                df5 = pd.DataFrame(df_data5, columns=columns5)
                columns6 = ['time']
                df6 = pd.DataFrame(df_data6, columns=columns6)
                df = pd.concat([df1,df2,df3,df4,df5,df6], axis=1)
                dfs.append(df)
                df1 = 0
                df2 = 0
                df3 = 0
                df4 = 0
                df5 = 0
                df6 = 0
        df = pd.concat(dfs)
        df = df[df.item.isin(valid_item_asin)]

        df = df.replace({'price':item_price, 'title':item_title})
        df['price'] = df['price'].astype(float)
        df['time'] = df['time'].astype(int)


        #user_df = pd.DataFrame(np.array(list(set(df['user']))), columns=['user'])
        #user_df.to_csv(output_category_meta_user_name, index=False)
        #index_user_dict = df['user'].to_dict()
        #user_index_dict = {v:k for k, v in index_user_dict.items()}
        #index_user_dict = None

        #valid_asin = item_price.keys()
        #df = df[df.price.isin(valid_asin)]


        df = df.sort(['user'])
        df.to_csv(output_category_name[ix], index=False, encoding='utf-8')

        #过滤掉只买过一个的用户
'''
        tmp_df = df.groupby('user').count()
        tmp_df = tmp_df[tmp_df.price > 1]
        valid_user = set(tmp_df.index)
        df = df[df.user.isin(valid_user)]

        df = df.sort_values(by='user')
        df.to_csv(output_category_review_name[ix], index=False, encoding='utf-8')
'''


if __name__ == '__main__':
    main()
