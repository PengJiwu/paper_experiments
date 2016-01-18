#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import json
import numpy as np
import pandas as pd
import os

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    #lowest_user_reviews = 1
    #lowest_product_reviews = 1
    input_dir = '/Users/tao/Documents/coding/dataset/workplace/'
    output_dir = input_dir
    output_meta = '_meta.csv'
    output_reviews = '.review'
    output_products = '.product'
    piece_len = 600000

    #for file in os.listdir(input_dir):
    for ijlik in range(1):
        '''
        if file[0] == '.' or os.path.isdir(file):
            continue

        dfs = []
        with open(input_dir+file, 'r') as f:

            data = f.readlines()
            pieces = [data[x:x+piece_len] for x in xrange(0, len(data), piece_len)]
            for piece in pieces:
                piece_json = []
                for line in piece:
                    try:
                        json_data = eval(line)
                    except ValueError:
                        continue
                    piece_json.append(json_data)

                df_data1 = [[x['reviewerID']] for x in piece_json]
                df_data2 = [[x['asin']] for x in piece_json]
                df_data3 = [[x['overall']] for x in piece_json]
                df_data4 = [[x['reviewText']] for x in piece_json]
                columns1 = ['user']
                df1 = pd.DataFrame(df_data1, columns=columns1)
                columns2 = ['item']
                df2 = pd.DataFrame(df_data2, columns=columns2)
                columns3 = ['rate']
                df3 = pd.DataFrame(df_data3, columns=columns3)
                columns4 = ['review']
                df4 = pd.DataFrame(df_data4, columns=columns4)
                df = pd.concat([df1,df2,df3,df4], axis=1)
                dfs.append(df)
                df1 = 0
                df2 = 0
                df3 = 0
                df4 = 0

            review_data = json.load(f)
            df_data1 = [[x['reviewerID']] for x in review_data]
            df_data2 = [[x['asin']] for x in review_data]
            df_data3 = [[x['overall']] for x in review_data]
            df_data4 = [[x['reviewText']] for x in review_data]
            columns1 = ['user']
            df1 = pd.DataFrame(df_data1, columns=columns1)
            columns2 = ['item']
            df2 = pd.DataFrame(df_data2, columns=columns2)
            columns3 = ['rate']
            df3 = pd.DataFrame(df_data3, columns=columns3)
            columns4 = ['review']
            df4 = pd.DataFrame(df_data4, columns=columns4)
            df = pd.concat([df1,df2,df3,df4], axis=1)

        #df = pd.concat(dfs)


        #Filtering
        #df = df[df.review != '']
        #filter_df = df.groupby('user').agg('count')
        #wanted = filter_df[filter_df['rate'] >= lowest_user_reviews].index
        #df = df[df.user.isin(wanted)]
        #filter_df = df.groupby('item').agg('count')
        #wanted = filter_df[filter_df['rate'] >= lowest_product_reviews].index
        #df = df[df.item.isin(wanted)]
        #filter_df = 0
        #wanted = 0
        '''
        file = 'review_shirts'
        df = pd.read_csv(input_dir+file)

        df_group_list = []
        f0 = open(output_dir + file + output_reviews, 'w')
        with open(output_dir + file + output_products, 'w') as f1:
            for item_value, item_df in df.groupby('item'):
                df_group_list.append(item_df)
                f1.write(item_value + '\t' + str(item_df.shape[0]) + '\t' + 'url\n')
                for review in item_df['text']:
                    f0.write('<DOC>\n')
                    f0.write(review)
                    f0.write('\n</DOC>\n')
                    f1.write('\t<DOC>\n\t')
                    f1.write(review)
                    f1.write('\n\t</DOC>\n')
        f0.close()
        df = 0

        df_group = pd.concat(df_group_list)
        df_group.to_csv(output_dir + file + output_meta)
        df_group = 0



if __name__ == '__main__':
    main()
