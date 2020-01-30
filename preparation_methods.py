from pull_stock import pull_stock
from pull_trend import pull_keywords_trend
from supports import general_path
from datetime import datetime
from datetime import timedelta
import os
import pathlib


def pull_all_inputs_date(last_day,
                         stock_name,
                         keywords,
                         study_name,
                         number_of_data_sets=20,
                         number_of_weeks_per_set=4):
    """
    here is the place the gather all the study DATA!!!
    to feed the analyze kit
    :param number_of_training_sets:
    :param days_frequency:
    :param stock_name:
    :param keyworks:
    :return:
    """
    # this thing one shot pulls everying
    # they why a progress bar is needed here
    # suppose we pull 20 sets
    # it will form a folder
    # and generates all the input data

    # let's suppose the start date is today
    # make last_day into a datetime obj
    last_day = datetime.strptime(last_day, '%Y-%m-%d')

    # lets form a folder to store all the data before pull
    # lets form a project file directory
    # we need a general path
    # lets create a study name, with study creation date
    # later the analyze kit can load a study.
    # add a date stemp
    directory = os.path.join(general_path(), f'{study_name}_on_{datetime.today()}')
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    for set_number in range(number_of_data_sets):
        print(f'pulling data for set number {set_number}')
        # now lets form a folder
        # name it xx studies
        # get each one time frame for x
        # get each one time frame for y
        trend_end_date = last_day - timedelta(days=7 * (set_number + 1))  # +1 because give some time to the stock
        trend_start_date = trend_end_date - timedelta(days=7 * number_of_weeks_per_set - 1)

        # time to pull y
        stock_end_date = last_day - timedelta(days=7 * set_number)
        stock_start_date = stock_end_date - timedelta(days=6)

        # conversion both to string so that we can pass to the pytrend.. (date to string.. string to date)
        trend_end_date = trend_end_date.strftime('%Y-%m-%d')
        trend_start_date = trend_start_date.strftime('%Y-%m-%d')
        stock_end_date = stock_end_date.strftime('%Y-%m-%d')
        stock_start_date = stock_start_date.strftime('%Y-%m-%d')

        # get the save path
        save_path_trend = os.path.join(directory, f'trend_for_stock_end_at_{stock_end_date}.csv')
        save_path_stock = os.path.join(directory, f'stock_for_stock_end_at_{stock_end_date}.csv')

        # form the time_frame
        timeframe = f'{trend_start_date} {trend_end_date}'
        pull_keywords_trend(keywords_list=keywords,
                            time_frame=timeframe,
                            save_path=save_path_trend,
                            relative_to_each_other=False)
        # pull stock
        pull_stock(stock_name=stock_name, start=stock_start_date, end=stock_end_date, save_path=save_path_stock)

    return 0
