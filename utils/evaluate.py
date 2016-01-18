#-*- coding: utf-8 -*-
import math

def precision(model_result, real_result):
    """计算模型得到的推荐结果的准确率
    Args:
        model_result:以(user_id,item_id)为元素的集合
        real_result:以(user_id,item_id)为元素的集合
    Return:
        返回一个float，精确到小数点后四位的精确度
    """
    hit_items = len(model_result.intersection(real_result))
    predict_items = len(model_result)
    tmp = "%.4f" % ((hit_items+0.0) / predict_items)
    return float(tmp)

def recall(model_result, real_result):
    """计算模型得到的推荐结果的召回率
    Args:
        model_result:以(user_id,item_id)为元素的集合
        real_result:以(user_id,item_id)为元素的集合
    Return:
        返回一个float，精确到小数点后四位的召回率
    """
    hit_items = len(model_result.intersection(real_result))
    real_users_items = len(real_result)
    tmp = "%.4f" % ((hit_items+0.0) / real_users_items)
    return float(tmp)

def f_score(model_result, real_result):
    """求F值
    Args:
        model_result:以(user_id,item_id)为元素的集合
        real_result:以(user_id,item_id)为元素的集合
    Return:
        返回一个float，精确到小数点后四位的F1 score
    """
    p = precision(model_result, real_result)
    r = recall(model_result, real_result)
    tmp = 0
    if not (p == 0 or r == 0):
        tmp = "%.5f" % (p * r * 2 / (p + r))
    return p, r, float(tmp)

def f_score_of_feature(model_result, real_result):
    p_sum = 0.0
    r_sum = 0.0
    f_sum = 0.0
    for ix, val in real_result.iteritems():
        model_u = model_result[ix[0]]
        p, r, f = f_score(model_u, val)
        p_sum += p
        r_sum += r
        f_sum += f

    size = len(real_result)
    print f_sum, size
    avg_p = "%.5f" % (p_sum / size)
    avg_r = "%.5f" % (r_sum / size)
    avg_f = "%.5f" % (f_sum / size)
    return avg_p, avg_r, avg_f