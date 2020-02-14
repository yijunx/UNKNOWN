from pytrends.request import TrendReq
from datetime import datetime
from datetime import timedelta
import pandas as pd
import sys
import os
import pathlib
from supports import general_path


def pull_keywords_trend(keywords_list,
                        time_frame,
                        geo='US',
                        save_folder=None,
                        relative_to_each_other=True):
    """

    :param keywords_list: up to 5
    :param time_frame: Specific dates, 'YYYY-MM-DD YYYY-MM-DD' example '2016-12-14 2017-01-25'
    :param geo:
    :param save_folder: the path to save, optional
    :param relative_to_each_other:

    # notes on the keywords:
    last time used:
    ['AMGN','CELG','BIIB','GILD','REGN']

    # notes in the timeframe

    - Date to start from

    - Defaults to last 5yrs, 'today 5-y'.

    - Everything 'all'

    - Specific dates, 'YYYY-MM-DD YYYY-MM-DD' example '2016-12-14 2017-01-25'

    - Specific datetimes, 'YYYY-MM-DDTHH YYYY-MM-DDTHH' example '2017-02-06T10 2017-02-12T07'

    Note Time component is based off UTC
    Current Time Minus Time Pattern:

    By Month: 'today #-m' where # is the number of months from that date to pull data for

    For example: 'today 3-m' would get data from today to 3months ago
    NOTE Google uses UTC date as 'today'
    Seems to only work for 1, 2, 3 months only
    Daily: 'now #-d' where # is the number of days from that date to pull data for

    For example: 'now 7-d' would get data from the last week
    Seems to only work for 1, 7 days only
    Hourly: 'now #-H' where # is the number of hours from that date to pull data for

    For example: 'now 1-H' would get data from the last hour
    Seems to only work for 1, 4 hours only

    35 weeks is by day
    50 weeks is by week

    :return: DataFrame
    """
    # , proxies=['https://35.201.123.31:880', ]
    pytrends = TrendReq(hl='en-US', tz=360)  # tz is time zone offset in minutes
    if relative_to_each_other:
        pytrends.build_payload(keywords_list, cat=0, timeframe=time_frame, geo=geo, gprop='')
        interest_over_time_df = pytrends.interest_over_time()
    else:
        print('means that the keywords will be pulled one by one.. with each one has a 100, in the period')
        interest_over_time_df = pd.DataFrame()
        for keyword in keywords_list:
            pytrends.build_payload([keyword], cat=0, timeframe=time_frame, geo=geo, gprop='')
            if len(interest_over_time_df) == 0:
                interest_over_time_df = pytrends.interest_over_time()
            else:
                interest_over_time_df[keyword] = pytrends.interest_over_time()[keyword]

    # now lets check if it is by week or by day
    a_delta = interest_over_time_df.index[1] - interest_over_time_df.index[1]
    if a_delta.days > 1:
        by_day = False
        print('This trend is by week')
    else:
        by_day = True
        print('This trend is by day')

    # now lets get the file name for the pulled trend
    file_name = f'{"_".join(keywords_list)}_{"by_day" if by_day else "by_week"}.csv'
    print(f'trend file name is {file_name}')
    if save_folder:
        interest_over_time_df.to_csv(os.path.join(save_folder, file_name))
    print(f'Total lines: {len(interest_over_time_df)}')
    print(interest_over_time_df)
    return interest_over_time_df


if __name__ == "__main__":
    # date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

    # the system input position starts with 1
    number_of_weeks = int(sys.argv[1])  # make the string into an integer

    # weeks offset is the send offset number
    # dont use weeks offset any mmore
    input_date = sys.argv[2]  # which is string
    end_date = datetime.strptime(input_date, '%Y-%m-%d')

    # get the start day
    start_date = end_date - timedelta(days=number_of_weeks * 7)

    # conversion both to string so that we can pass to the pytrend.. (date to string.. string to date)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # form the time_frame
    time_frame = f'{start_date} {end_date}'

    # print some basic information
    print()  # leave some space here
    print(f'Pulling {number_of_weeks} weeks data, end at {input_date}')
    print(f'End is {input_date}')
    print(f'Start date is {start_date}')

    # do the pull here, and save, need to make it auto run, it is good to do the run at monday 6 to 7 pm
    # before the decision is made
    # generate the save path

    # pulled_at_{datetime.today().date()}_end_at_{input_date}_for_{number_of_weeks}_weeks.csv
    keywords_list = ['AMGN', 'VRTX', 'BIIB', 'GILD', 'REGN', 'ILMN', 'ALXN', 'SGEN', 'INCY']
    save_path = general_path()

    # composition names
    # ['AMGN', 'VRTX', 'BIIB', 'GILD', 'REGN', 'ILMN']
    # more composition names
    #
    pull_keywords_trend(keywords_list=keywords_list,
                        time_frame=time_frame,
                        save_folder=save_path,
                        relative_to_each_other=False)


