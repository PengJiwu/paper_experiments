#-*- coding:utf-8 -*-

__author__ = 'tao'

from RRCP.utils import get_top_columns_index

def _change_X_with_features(X, features):
    init_set = set(range(X.shape[1]))
    for idx, row in X.iterrows():
        #if idx % 100 == 0:
            #print idx
        zeros_set = init_set.difference(features[idx])
        for cdx in zeros_set:
            row[cdx] = 0

def recommend_features(paper_symbol_U_1, paper_symbol_V, feature_k):
    X = paper_symbol_U_1.dot(paper_symbol_V.T)
    features = get_top_columns_index(X, feature_k)

    result = {}

    for idx in range(len(features)):
        result[idx] = set(features[idx])

    return result

def recommend_items(paper_symbol_U_1, paper_symbol_U_2, paper_symbol_V, paper_symbol_H_1, paper_symbol_H_2, train_buy_set, alpha,  feature_k, item_k):
    X = paper_symbol_U_1.dot(paper_symbol_V.T)
    Y = paper_symbol_U_2.dot(paper_symbol_V.T)
    A = paper_symbol_U_1.dot(paper_symbol_U_2.T) + paper_symbol_H_1.dot(paper_symbol_H_2.T)

    features = get_top_columns_index(X, feature_k)

    #print features

    _change_X_with_features(X, features)

    #print 'change OK'

    tmp1 = alpha * X.dot(Y.T) / (feature_k * 5)
    tmp1.index = range(tmp1.shape[0])
    tmp1.columns = range(tmp1.shape[1])
    tmp2 = (1 - alpha) * A
    tmp2.index = range(tmp2.shape[0])
    tmp2.columns = range(tmp2.shape[1])
    #print 'tmp1', sorted(tmp1.ix[0,:], reverse=True)[:8]
    #print 'tmp2', sorted(tmp2.ix[0,:], reverse=True)[:8]
    score = tmp1.add(tmp2, fill_value=0)
    #print 'score', score.ix[0,:5]

    #score.to_csv('score.csv', index=False)
    #print score.ix[0,0]

    items = get_top_columns_index(score, score.shape[1])

    #print items[0]

    result = set([])

    for idx in range(len(items)):
        val = items[idx]
        user_items_num = 0
        for cdx in val:
            potentional_ui = (idx, cdx)
            #if not potentional_ui in train_buy_set:
            result.add(potentional_ui)
            user_items_num += 1
            if user_items_num >= item_k:
                break

    return result
