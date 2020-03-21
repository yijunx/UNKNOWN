"""

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""


def general_path():
    return r'/Users/yijunxu/Dropbox/UNKNOWN_RESULTS'


def find_kw_based_on_cat_name(kw_set_name):
    return kw_dicts()[kw_set_name]


def kw_dicts():

    # this guy should have all thing there .... is it????
    # should have all the things that,
    # each step of keyworkds should have a kw_set_name (which is the key of the dictionary)
    kw_dict = {'debt': 'debt',
               'compare_group': 'apple_pear_root bear',
               'debt_and_others': 'default_derivatives_debt_credit_crisis_gold price',
               'economics': 'inflation_housing_investment_travel_unemployment'}


    return kw_dict
