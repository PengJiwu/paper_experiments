#-*- coding:utf-8 -*-
#!/usr/bin/python

__author__ = 'tao'

import json
import numpy as np
import pandas as pd
import sys

def main():
    input = '/Users/tao/Documents/coding/dataset/dianping/reviews.json'
    output_reviews = '/Users/tao/Documents/coding/dataset/dianping/reviews.doc'
    output_meta = '/Users/tao/Documents/coding/dataset/dianping/reviews_metadata.pkl'

    result = []
    with open(input, 'r') as f:
        for line in f.readlines():
            try:
                json_data = json.loads(line)
            except ValueError:
                continue
            item = {'userId':json_data['userId'],'restId':json_data['restId'],'rate':json_data['rate'],'flavor':json_data['flavor'], 'environment':json_data['environment'],'service':json_data['service'],'content':json_data['content']}
            result.append(item)

    meta = [[x['userId'], x['restId'], x['rate'], x['flavor'], x['environment'], x['service']] for x in result]
    reviews = [x['content'] for x in result]

    df_data = np.array(meta)
    columns = ['userId', 'restId', 'rate', 'flavor', 'environment', 'service']
    df = pd.DataFrame(df_data, columns=columns)
    df.to_pickle(output_meta)

    reload(sys)
    sys.setdefaultencoding('utf-8')
    with open(output_reviews, 'w') as f:
        for review in reviews:
            f.write('<DOC>\n')
            f.write(review)
            f.write('</DOC>\n')



if __name__ == '__main__':
    main()