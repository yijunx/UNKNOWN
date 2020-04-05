"""

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
from datetime import timedelta
from datetime import datetime
from pull_trend import pull_keywords_trend


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
               'economics': 'inflation_housing_investment_travel_unemployment',
               'economics2': 'housing_investment_travel_unemployment_inflation',
               'newly_discovered': '',
               'bio_stuff_stock_names_only': '',
               'bio_stuff_meaningful_stuff': ''}


    return kw_dict


# def pull_keywords_trend(keywords_list,
#                         keyword_short_name,
#                         time_frame,
#                         geo='US',
#                         save_folder=None,
#                         relative_to_each_other=True):

def pull_trend_weekly(kw_small_name, number_of_weeks=200):

    relative_to_each_other = False

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=number_of_weeks * 7)

    # conversion both to string so that we can pass to the pytrend.. (date to string.. string to date)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # form the time_frame
    time_frame = f'{start_date} {end_date}'

    pull_keywords_trend(kw_dicts()[kw_small_name].split('_'), kw_small_name, time_frame=time_frame,
                        save_folder=general_path(), relative_to_each_other=relative_to_each_other)


if __name__ == '__main__':
    pull_trend_weekly('economics2')
