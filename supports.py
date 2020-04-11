"""

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
from datetime import timedelta
from datetime import datetime
from pull_trend import pull_keywords_trend
from pull_stock import pull_stock
import os


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


def pull_stock_nicely(stock_name, number_of_weeks=200):
    # the system input position starts with 1
    # weeks offset is the send offset number
    # dont use weeks offset any mmore

    end_date = datetime.now().date()

    # get the start day
    start_date = end_date - timedelta(days=number_of_weeks * 7)

    # conversion both to string so that we can pass to the pytrend.. (date to string.. string to date)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # form the time_frame
    time_frame = f'{start_date} {end_date}'

    # print some basic information
    print()  # leave some space here
    print(f'Pulling {number_of_weeks} weeks data, end at {end_date}')
    print(f'End is {end_date}')
    print(f'Start date is {start_date}')

    # do the pull here, and save, need to make it auto run, it is good to do the run at monday 6 to 7 pm
    # before the decision is made
    # generate the save path
    stock_name = 'SPY'
    print(f"stock file name is {stock_name}_end_at_{end_date}_for_{number_of_weeks}_weeks.csv")
    save_path = os.path.join(general_path(),
                             f'{stock_name}.csv')

    pull_stock(stock_name=stock_name, save_path=save_path, start=start_date, end=end_date)


if __name__ == '__main__':
    pull_trend_weekly('economics2')
