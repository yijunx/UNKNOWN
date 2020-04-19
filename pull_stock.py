import yfinance as yf
from datetime import timedelta
from datetime import datetime
import sys
from supports import general_path
import os


def pull_stock(stock_name, start, end, save_path=None):
    df = yf.download(stock_name, start=start, end=end)
    if save_path:
        df.to_csv(save_path)
    print('last 20 rows')
    print(df.tail(20))
    return df


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
    print(f"stock file name is {stock_name}_end_at_{end_date}_for_{number_of_weeks}_weeks.csv")
    save_path = os.path.join(general_path(),
                             f'{stock_name}.csv')

    pull_stock(stock_name=stock_name, save_path=save_path, start=start_date, end=end_date)

