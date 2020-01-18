from pytrends.request import TrendReq
import os


def pull_keywords_trend(keywords_list, time_frame, geo='US', save_folder_link=None):
    """

    :param keywords_list: up to 5
    :param time_frame: Specific dates, 'YYYY-MM-DD YYYY-MM-DD' example '2016-12-14 2017-01-25'
    :param geo:
    :param save_folder_link:

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
    :return:
    """
    pytrends = TrendReq(hl='en-US', tz=360)  # tz is time zone offset in minutes
    pytrends.build_payload(keywords_list, cat=0, timeframe=time_frame, geo=geo, gprop='')
    interest_over_time_df = pytrends.interest_over_time()

    # now we need to save them, instead of pulling them all the time
    # we need to save the them as a scheduled task, run every week / day depending on the training...
    # let's try day
    # print(interest_over_time_df.tail())
    if save_folder_link:
        interest_over_time_df.to_csv(os.path.join(save_folder_link, f'haha.csv'))

    return interest_over_time_df

if __name__ == "__main__":
    pull_keywords_trend(keywords_list=['apple', 'pear'],
                        time_frame='today 5-y',
                        save_folder_link=r'/Users/yijunxu/Dropbox/UNKNOWN_RESULTS')


