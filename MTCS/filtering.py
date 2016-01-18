__author__ = 'tao'

import pandas as pd

a = pd.read_csv('/Users/tao/Documents/coding/dataset/workplace/phones.csv')
b = a.groupby('user').count()
b = b[b.item >= 3]
vu = set(b.index)
a = a[a.user.isin(vu)]
a.to_csv('/Users/tao/Documents/coding/dataset/workplace/phones3.csv', index=False)