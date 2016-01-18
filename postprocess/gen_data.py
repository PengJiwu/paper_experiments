#coding = gbk
from __future__ import division
import math
import pandas as pd
import numpy as np
import ast

class gen_data():
    def __init__(self):
        print 'init'
    def load_data(self):
        file = '../Data/review/DC_feature_opinion/DC.txt'
        input = open(file, 'r')
        self.all_reviews = np.array(input.read().split('</DOC>'))
    def generate_data(self):
        records = []
        num = 0
        for line in self.all_reviews:
            print num
            num += 1
            if line != '':
                array_f = []
                array_1 = line.split('\n')
                for p in array_1:
                    array_f += p.split('\t')
                array_f = [i for i in array_f if i != '']
                feature = array_f[6:]
                fs = []
                for l in feature:
                    f = l.split(',')[0].replace('[', '').strip()
                    a = l.split(',')[2].strip()
                    b = l.split(',')[4].replace(']', '').strip()
                    if (str(a) == '1' and str(b) == 'N') or (str(a) == '-1' and str(b) == 'Y'):
                        s = 1
                    else:
                        if str(a) == '0':
                            s = 0
                        else:
                            s = -1
                    fs.append([f, s])
                print [array_f[1], array_f[2], fs, array_f[3]]
                records.append([array_f[1], array_f[2], fs, array_f[3]])

        t = pd.DataFrame(records)
        t.to_csv('v3_records')
    def ui_statics(self, filename):
        file = filename
        data = pd.read_csv(file)
        u_sta = dict()
        i_sta = dict()
        for line in data.values:
            user = line[1]
            item = line[2]
            if u_sta.has_key(user):
                u_sta[user] += 1
            else:
                u_sta[user] = 1
            if i_sta.has_key(item):
                i_sta[item] += 1
            else:
                i_sta[item] = 1
        return u_sta, i_sta
    def clear_data(self):
        file = 'v3_records'
        data = pd.read_csv(file)
        u_sta, i_sta = self.ui_statics(file)
        new_file = []
        for line in data.values:
            print line
            user = line[1]
            item = line[2]
            if u_sta[user] > 10 and i_sta[item] > 5:
                new_file.append(line[1:])
        t = pd.DataFrame(new_file)
        t.to_csv('v3_dense_records')
    def dense_ui(self):
        file = 'v3_dense_records'
        data = pd.read_csv(file)
        users = data['user']
        items = data['item']
        features = []
        for f in data['feature']:
            l = ast.literal_eval(f)
            features.extend([x[0] for x in l])

        t = pd.DataFrame(list(set(users)))
        t.to_csv('v3_users')
        t = pd.DataFrame(list(set(items)))
        t.to_csv('v3_items')
        t = pd.DataFrame(list(set(features)))
        t.to_csv('v3_features')
    def user2uid(self, user):
        file = 'v3_users'
        data = pd.read_csv(file)
        users = np.array(data.values[:,1])
        index = np.argwhere(users == user)
        return index[0][0]
    def item2iid(self, item):
        file = 'v3_items'
        data = pd.read_csv(file)
        items = np.array(data.values[:, 1])
        index = np.argwhere(items == item)
        return index[0][0]
    def feature2fid(self, feature):
        file = 'v3_features'
        data = pd.read_csv(file)
        features = np.array(data.values[:, 1])
        index = np.argwhere(features == feature)
        return index[0][0]
    def format_records(self):
        file = 'v3_dense_records'
        data = pd.read_csv(file)
        users = pd.read_csv()
        results = []
        ix = 0
        for line in data.values:
            ix += 1
            if ix % 1000 == 0:
                print ix
            #print line[1], line[2], line[3], line[4]

            uid = self.user2uid(line[1])
            iid = self.item2iid(line[2])
            fid = []
            for f in eval(line[3]):
                fid.append([self.feature2fid(f[0]), f[1]])
            #print [uid, iid, str(fid), line[4]]
            results.append([uid, iid, str(fid), line[4]])
        t = pd.DataFrame(results)
        t.to_csv('v3_formatted_records')
    def gen_train_and_test(self):
        file = 'v3_formatted_records'
        data = pd.read_csv(file)
        users_all = np.array(data.values)[:,1]
        users_set = list(set(users_all))
        users_records_num = [list(users_all).count(i) for i in range(len(users_set))]
        train_num = [int(i*0.7) for i in users_records_num]
        user_num = np.zeros(len(users_set))
        train = []
        test = []
        for line in data.values:
            uid = line[1]
            if user_num[uid] <= train_num[uid]:
                user_num[uid] += 1
                train.append(line[1:])
            else:
                test.append(line[1:])
        t = pd.DataFrame(train)
        t.to_csv('v3_train_records')
        t = pd.DataFrame(test)
        t.to_csv('v3_test_records')
    def feature_rating(self):
        file = 'v3_train_records'
        data = pd.read_csv(file, dtype='str')
        item_rating = []
        i = 0
        for line in data.values:
            print i
            i += 1
            iid = line[2]
            new_feature = line[3]
            if len(item_rating) == 0:
                item_rating = [[iid, new_feature]]
            else:
                if iid in np.array(item_rating)[:, 0]:
                    index = np.argwhere(np.array(item_rating)[:,0] == iid)[0][0]
                    features = eval(item_rating[index][1])
                    for f in eval(new_feature):
                        if f[0] in np.array(features)[:, 0]:
                            fi = np.argwhere(np.array(features)[:,0] == f[0])[0][0]
                            features[fi][1] += f[1]
                        else:
                            features.append(f)
                    item_rating[index][1] = str(features)
                else:
                    pre_feature = eval(new_feature)
                    post_feature = []
                    for pre_f in pre_feature:
                        if len(post_feature) == 0:
                            post_feature.append(pre_f)
                        else:
                            if pre_f[0] in np.array(post_feature)[:, 0]:
                                pre_fi = np.argwhere(np.array(post_feature)[:,0] == pre_f[0])[0][0]
                                post_feature[pre_fi][1] += pre_f[1]
                            else:
                                post_feature.append(pre_f)
                    item_rating.append([iid, str(post_feature)])
        result_rating = []
        for line in item_rating:
            print line
            iid = line[0]
            f_ratings = eval(line[1])
            result_f = [[i[0], (1+4.0/(1.0 + math.exp(-i[1])))] for i in f_ratings]
            result_rating.append([iid, str(result_f)])
            print [iid, str(result_f)]

        t = pd.DataFrame(result_rating)
        t.to_csv('v3_feature_rating')
    def gen_test_data(self):
        file = 'v3_test_records'
        data = pd.read_csv(file, dtype='str')
        result = []
        for line in data.values:
            uid = line[1]
            iid = line[2]
            fids = line[3]
            key = uid + '@' + iid
            if len(result) == 0:
                result.append([key, fids])
            else:
                if key in np.array(result)[:, 0]:
                    index = np.argwhere(np.array(result)[:, 0] == key)[0][0]
                    result[index][1] = str(eval(result[index][1])+eval(fids))
                else:
                    result.append([key, fids])
        t = pd.DataFrame(result)
        t.to_csv('v3_test_data')
    def gen_user_favor_feature(self):
        file = 'v3_train_records'
        user_file = 'v3_users'
        feature_file = 'v3_features'

        data = pd.read_csv(file, dtype='str')
        user_data = pd.read_csv(user_file, dtype='str')
        feature_data = pd.read_csv(feature_file, dtype='str')

        result = np.zeros((len(user_data), len(feature_data)))
        for line in data.values:
            user = line[1]
            feature = eval(line[3])
            for f in feature:
                result[int(user)][f[0]] += 1

        r = []
        for line in result:
            index = np.argsort(line)[-1:-6:-1]
            r.append(index)
        t = pd.DataFrame(r)
        t.to_csv('v3_user_favor_feature')
    def gen_score(self):
        file = 'v3_train_records'
        data = pd.read_csv(file, dtype='str')
        ui_scores = []
        for line in data.values:
            uid = line[1]
            iid = line[2]
            key = uid + '@' + iid
            score = int(line[4])
            print uid, iid, score
            if len(ui_scores) == 0:
                ui_scores=[[key, str([score])]]
            else:
                if key in np.array(ui_scores)[:, 0]:
                    index = np.argwhere(np.array(ui_scores)[:, 0] == key)[0][0]
                    ui_scores[index][1] = str(eval(ui_scores[index][1])+[score])
                else:
                    ui_scores.append([key, str([score])])
        result = []
        for line in ui_scores:
            print line
            key = line[0]
            scores = eval(line[1])
            s = np.array(scores).sum() / len(scores)
            print [key, s]
            result.append([key, s])
        t = pd.DataFrame(result)
        t.to_csv('v3_ui_score')


if __name__ == '__main__':
    g = gen_data()
    #g.load_data()
    #g.generate_data()
    g.dense_ui()

    g.format_records()
    #g.gen_user_feature_dict()
    #g.gen_dense_users()
    #g.gen_dense_items()
    #g.gen_dense_features()
    #g.gen_dense_final_records()
    #g.gen_train_and_test()
    #g.add_rating_and_sentiment()
    #print g.uid2user(10)
    #print g.iid2item(10)
    #g.feature_rating()
    #g.clear_items()
    #g.clear_data()
    #g.dense_ui()
    #print g.ui_statics('v2_dense_records')
    #g.gen_test_data()
    #g.gen_user_favor_feature()
    #g.gen_score()



