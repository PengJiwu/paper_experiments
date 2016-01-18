__author__ = 'tao'

import pandas as pd
from collections import Counter

df = pd.read_csv('/Users/tao/Documents/coding/dataset/workplace/phones.csv')
df = df.groupby('user').count()
users = df['item']
c = Counter(users)
print c.most_common()