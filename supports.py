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
    kw_dict = {'SPY-debt': 'debt',
               'SPY-compare_group': 'apple_pear_root bear',
               'SPY-debt_and_others': 'default_derivatives_debt_credit_crisis_gold price',
               'SPY-economics': 'inflation_housing_investment_travel_unemployment_bear market_bull market_crude oil_coronavirus_pandemic',
               'SPY-economics2': 'housing_investment_travel_unemployment_inflation',
               # 'newly_discovered': '',
               'BIB-bio_stuff_stock_names_only': f"{'_'.join(['AMGN','CELG','BIIB','GILD','REGN', 'VRTX'])}",
               'BIB-bio_stuff_meaningful_stuff': 'drug price_drug approval_FDA_Biotech industry_insulin_health care'}

    return kw_dict


# def pull_keywords_trend(keywords_list,
#                         keyword_short_name,
#                         time_frame,
#                         geo='US',
#                         save_folder=None,
#                         relative_to_each_other=True):

