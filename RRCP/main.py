__author__ = 'tao'

import pandas as pd

import Construction
import MyGaussian


def main():
    a = pd.read_csv('../../../dataset/workplace/filter_phones_format.csv')
    b = {'phones':a}
    price_df = Construction.get_user_category_buy_price(b)
    param_mu, param_sigm = MyGaussian.gaussian_curve_fit(price_df)
    print min(param_sigm['phones']), max(param_sigm['phones'])
if __name__ == '__main__':
    main()