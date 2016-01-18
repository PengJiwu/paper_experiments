#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import numpy as np
import pandas as pd
import json
import pickle

def main():
    input_dir = '/Users/tao/Documents/coding/dataset/workplace/'
    #input_dir = '/home/qzlab/tao_workstation/price/'
    output_dir = input_dir
    piece_len = 500000

    meta_categories = ['Clothing_Shoes_and_Jewelry2']
    review_categories = ['Clothing_Shoes_and_Jewelry2']
    input_category_meta_name = []
    input_category_review_name = []
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
            item_price = {}
            item_title = {}
            i = 0
            for item in f.readlines():
                if i % 10000 == 0:
                    print i
                i += 1
                item_dict = eval(item)
                if 'price' in item_dict and 'title' in item_dict:
                    valid_items.append(item_dict)
            '''
            valid_items = json.load(f)
            '''

        df_data = np.array([[x['asin'], x['title'], float(x['price'])] for x in valid_items])
        column_name = ['item', 'title', 'price']
        df = pd.DataFrame(df_data, columns=column_name)
        item_price = df.set_index('item')['price'].to_dict()
        item_title = df.set_index('item')['title'].to_dict()
        print 'meta lenght: ', len(df)
        valid_item_asin = set(df['item'])
        #Fileter
        #df = df[df.price >= 5]
        #df = df[df.price < 300]
        #df.to_csv(output_category_meta_item_name[ix], index=False, encoding='utf-8')



        dfs = []
        print 'meta process over and review begin'
        with open(input_category_review_name[ix], 'r') as f:
            data = [eval(x) for x in f.readlines()]
            data = [x for x in data if x['asin'] in valid_item_asin]
            pieces = [data[x:x+piece_len] for x in xrange(0, len(data), piece_len)]
            i = 0
            for piece_json in pieces:
                print 'piece of ', i
                i += 1

                df_data1 = [[x['reviewerID']] for x in piece_json]
                df_data2 = [[x['asin']] for x in piece_json]
                df_data3 = [item_title[x['asin']] for x in piece_json]
                df_data4 = [item_price[x['asin']] for x in piece_json]
                df_data5 = [[x['overall']] for x in piece_json]
                df_data6 = [[x['unixReviewTime']] for x in piece_json]
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


        '''
        with open('item_price.pkl', 'w') as f:
            pickle.dump(item_price, f)
        print 'price save over'
        with open('item_title.pkl', 'w') as f:
            pickle.dump(item_title, f)
        print 'title save over'
        df.to_csv(output_category_name[ix], index=False, encoding='utf-8')
        print 'df save over'
        '''

        df['price'] = df['price'].astype(float)
        df['time'] = df['time'].astype(int)

        tmp_df = df.groupby('user').count()
        tmp_df = tmp_df[tmp_df.item > 1]
        vu = set(tmp_df['user'])
        df = df[df.user.isin(vu)]

        df.to_csv(output_category_name[ix], index=False, encoding='utf-8')
        print 'output over'


if __name__ == '__main__':
    main()
