#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import numpy as np
import pandas as pd
import json

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    filepath = '/Users/tao/Documents/coding/dataset/AmazonPrice/'
    reviewpath = filepath + 'review_shirts.jl'
    outputpath = filepath + 'review_shirts.csv'

    with open(reviewpath, 'r') as f:
        review = json.load(f)

    #抓取的价格格式有 12 - $ 20 的形式，处理这种形式，取其平均值
    for ix in xrange(len(review)):
        prices = review[ix]['price'].split(' - ')
        if len(prices) == 2:
            low_price = float(prices[0])
            high_price = float(prices[1][1:])
            avg_price = round(low_price + (high_price - low_price)/2)
            review[ix]['price'] = str(int(avg_price))
        else:
            review[ix]['price'] = str(int(float(review[ix]['price'])))

    #将价格结合到评论里
    review = pd.DataFrame(review)

    review['rate'].astype(int)
    review['price'].astype(int)
    review.to_csv(outputpath, index=False)


if __name__ == '__main__':
    main()
