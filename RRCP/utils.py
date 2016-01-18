#-*- coding:utf-8 -*-

__author__ = 'tao'

import pandas as pd

def get_top_columns_index(X, top_k):
    row_num = X.shape[0]
    column_num = X.shape[1]
    if top_k > column_num:
        top_k = column_num

    #result = pd.Series([[] for _ in range(row_num)], index=range(row_num))
    result = []
    for idx, row in X.iterrows():
        result.append(sorted(range(column_num), key=lambda x: row[x], reverse=True)[:top_k])

    return result

