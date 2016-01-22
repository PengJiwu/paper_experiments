__author__ = 'tao'

import pandas as pd

a = pd.read_csv('/Users/tao/Documents/coding/dataset/workplace/phones.csv')
b = a.groupby('user').count()
b = b[b.item >= 5]
vu = set(b.index)

b = a.groupby('item').count()
b = b[b.user >= 5]
vi = set(b.index)

a = a[a.user.isin(vu)]
a = a[a.item.isin(vi)]
a.to_csv('/Users/tao/Documents/coding/dataset/workplace/phonesu5i5.csv', index=False)
