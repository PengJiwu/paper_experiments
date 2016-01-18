#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

from Construction import construction
from RRCP.utils import *

def main():
    path = '/Users/tao/Documents/coding/dataset/AmazonReviews/product_review15_category3_item4/'
    with open('result_amazon_with_1534', 'a') as f:
        for ix in [5]:
            print str(ix) + ' Begin'
            model_result, test_result = construction(path + str(ix) + '/')
            precision, recall, fscore = f_score_of_feature(model_result, test_result)

            feature = pd.read_csv(path + str(ix) + '/feature.csv')
            print feature.ix[list(model_result[0])]

            result = str(ix) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(fscore) + '\n'
            print result
            f.write(result)
        print str(ix) + ' End'

if __name__ == '__main__':
    main()