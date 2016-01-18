#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

from Construction import construction
from Compute import compute
from evaluate import f_score
from evaluate import f_score_of_feature
from recommender import recommend_features
from recommender import recommend_items

def _get_paramenters():
    r_plus_r_p = reversed([25, 50, 75, 100])
    lamda_x = 1
    lamda_y = 1
    lamda_u = 0.0003
    lamda_h = 0.0003
    lamda_v = 0.0003
    convergence = 0.00000005
    T = 200

    return r_plus_r_p, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v, convergence, T

def main():
    path = '/Users/tao/Documents/coding/dataset/YelpReviews/product_review5_category3_item4/'
    print 'yelp 10 f'
    with open('result_yelp_with_534_10f', 'a') as f:
        for ix in reversed([5, 6]):
            print str(ix) + ' Begin'
            paper_symbol_A, paper_symbol_X, paper_symbol_Y,  user_num, item_num, feature_num, train_buy_set, real_result, test_mention_feature = construction(path + str(ix) + '/')
            r_plus_r_p, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v, convergence, T = _get_paramenters()
            for sum in r_plus_r_p:
                paper_symbol_U_1, paper_symbol_U_2, paper_symbol_V, paper_symbol_H_1, paper_symbol_H_2 = compute(paper_symbol_A, paper_symbol_X, paper_symbol_Y,  user_num, item_num, feature_num, sum*0.4, sum * 0.6, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v, convergence, T)

                #print T, paper_symbol_U_1.ix[:5,:5], (paper_symbol_U_1.dot(paper_symbol_U_2.T) + paper_symbol_H_1.dot(paper_symbol_H_2.T)).ix[:5,:5]
                for alpha in [0]:#,1-0.1**2,1-0.1**3,1-0.1**4,1-0.1**5,1-0.1**6,1-0.1**7,1-0.1**8,1-0.1**9,1-0.1**10,1-0.1**11,1-0.1**12,1-0.1**13]:
                    #model_result = recommend_items(paper_symbol_U_1, paper_symbol_U_2, paper_symbol_V, paper_symbol_H_1, paper_symbol_H_2, train_buy_set, alpha, 50, 10)

                    model_result = recommend_features(paper_symbol_U_1, paper_symbol_V, 10)
                    precision, recall, f1_result = f_score_of_feature(model_result, test_mention_feature)
                    #precision, recall, f1_result = f_score(model_result, real_result)

                    result = str(ix) + '\t' + str(sum) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(f1_result) + '\n'
                    print result
                    f.write(result)
        print str(ix) + ' End'

if __name__ == '__main__':
    main()
