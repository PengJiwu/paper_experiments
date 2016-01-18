__author__ = 'tao'

import pandas as pd
from collections import Counter

df = pd.read_csv('/Users/tao/Documents/coding/dataset/workplace/phones.csv')
df = df.groupby('item').count()
users = df['price']
c = Counter(users)
d = c.most_common()
print sorted(d, key=lambda x: x[0])
