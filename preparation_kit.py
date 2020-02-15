import pandas as pd
from supports import general_path
import os
import numpy as np
from datetime import timedelta


def read_trend(trend_file_name):
    # pulled_at_2020-01-30_end_at_2020-1-11_for_35_weeks.csv
    df = pd.read_csv(os.path.join(general_path(), trend_file_name), index_col=0)
    df.index = pd.to_datetime(df.index)

    # now i need to scale the trend
    for col in df.columns:
        if col != 'isPartial':
            df[col] = df[col] / 100
    return df


def read_stock(stock_file_name):
    df = pd.read_csv(os.path.join(general_path(), stock_file_name), index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def first_pass_analysis(trend_file_name=None, stock_file_name=None):
    # not really used

    if trend_file_name is None:
        trend_file_name = 'pulled_at_2020-01-30_end_at_2020-1-11_for_35_weeks.csv'

    if stock_file_name is None:
        stock_file_name = 'stock_end_at_2020-1-30_for_52_weeks.csv'

    # read trend
    trend = read_trend(trend_file_name)

    # read stock
    stock = read_stock(stock_file_name)

    # merged_df
    # think about how to merge them
    # because there are days with only trends, no stock data, public holiday or weekends..
    trend_stock_df = trend.copy()
    for col in stock.columns:
        trend_stock_df[col] = stock[col]
    # let's include a plot here
    trend_columns = [x for x in trend.columns if x != 'isPartial']
    stock_columns = ['Open', 'Close', 'High', 'Low']

    trend_stock_df[trend_columns].plot()
    trend_stock_df[stock_columns].plot()

    # get the week
    # datetime.date(2010, 6, 16).isocalendar()[1]
    trend_stock_df['week'] = [x.date().isocalendar()[1] for x in trend_stock_df.index]
    print(trend_stock_df.head())

    return trend_stock_df


def give_week_number_to_dataframe(df):
    # now lets make them into weeks info

    df['week'] = [x.date().isocalendar()[1] for x in df.index]

    # so now lets chop into weeks
    df['week_diff'] = df.week.diff()
    df['week_diff'] = [x if x == 0 else 1.0 for x in df['week_diff']]
    df.loc[df.index[0], 'week_diff'] = 0.0

    df['week_number'] = df['week_diff'].cumsum()
    df['week_number'] = df['week_number'].apply(int)
    return df


def form_X_y_from_weekly_data(trend_file_name=None, stock_file_name=None, weeks_to_predict=4, predict_what=None):
    if trend_file_name is None:
        trend_file_name = 'pulled_at_2020-02-10_end_at_2020-2-7_for_100_weeks.csv'

    if stock_file_name is None:
        stock_file_name = 'stock_end_at_2020-2-7_for_100_weeks.csv'

    # read trend
    trend = read_trend(trend_file_name)

    # read stock
    stock = read_stock(stock_file_name)

    # pd.date_range('09-01-2013', '09-30-2013')
    # actually it is string from time
    stock_start_date = stock.index[0].strftime('%Y-%m-%d')
    stock_end_date = stock.index[-1].strftime('%Y-%m-%d')
    full_date_stock_df = pd.DataFrame(index=pd.date_range(start=stock_start_date,
                                                          end=stock_end_date))
    for col in stock.columns:
        full_date_stock_df[col] = stock[col]

    full_date_stock_df = give_week_number_to_dataframe(full_date_stock_df)

    week_summary = full_date_stock_df.groupby(['week_number']).agg({
        'week': 'count',
        'High': 'max',
        'Low': 'min',
    })

    week_summary = week_summary[week_summary.week >= 5]
    week_summary['Open_date'] = [full_date_stock_df.loc[full_date_stock_df.week_number == week_no, :].apply(pd.Series.first_valid_index).Open
                                 for week_no in week_summary.index]
    week_summary['Open'] = [full_date_stock_df.loc[date, 'Open'] for date in week_summary.Open_date]

    week_summary['Close_date'] = [full_date_stock_df.loc[full_date_stock_df.week_number == week_no, :].apply(pd.Series.last_valid_index).Close
                                  for week_no in week_summary.index]
    week_summary['Close'] = [full_date_stock_df.loc[date, 'Close'] for date in week_summary.Close_date]
    week_summary['close_open'] = week_summary.Close - week_summary.Open  # can be easily 1 or 0 it is binary
    week_summary['close_open'] = [0 if x > 0 else 1 for x in week_summary['close_open']]

    # now , lets clear based on the week summary to find out the x data frame
    # then we flatten x, then we can pass it to the solver

    inputs = []
    targets = []
    week_info = []

    for index, row in week_summary.iterrows():
        # date_mask = (trend.index > row.Open_date - timedelta(days=7 * weeks_to_predit)) & (trend)
        if len(trend.loc[row.Open_date - timedelta(days=7 * weeks_to_predict):row.Open_date, :]) >= weeks_to_predict:
            # lets get the date range...
            # date_mask = (data.index > start) & (data.index < end)
            # dates = data.index[date_mask]

            an_input = trend.loc[row.Open_date - timedelta(days=7 * weeks_to_predict):row.Open_date, :].drop(columns=['isPartial'])
            # make an_input to a list of numbers
            an_input = np.array(an_input).flatten()


            # let's use close minus open first
            a_target = row[predict_what]

            inputs.append(an_input)
            targets.append(a_target)
            week_info.append(row)

    return np.stack(tuple(inputs)), targets, week_info


def form_X_y_from_daily_data(trend_file_name=None, stock_file_name=None, weeks_to_predict=4, predict_what=None):

    if trend_file_name is None:
        trend_file_name = 'pulled_at_2020-01-30_end_at_2020-1-11_for_35_weeks.csv'

    if stock_file_name is None:
        stock_file_name = 'stock_end_at_2020-1-30_for_52_weeks.csv'

    # read trend
    trend = read_trend(trend_file_name)

    # read stock
    stock = read_stock(stock_file_name)

    # merged_df
    # think about how to merge them
    # because there are days with only trends, no stock data, public holiday or weekends..
    trend_stock_df = trend.copy()
    for col in stock.columns:
        trend_stock_df[col] = stock[col]
    # let's include a plot here
    trend_columns = [x for x in trend.columns if x != 'isPartial']
    stock_columns = ['Open', 'Close', 'High', 'Low']

    # trend_stock_df[trend_columns].plot()
    # trend_stock_df[stock_columns].plot()

    # get the week
    # datetime.date(2010, 6, 16).isocalendar()[1]
    trend_stock_df['week'] = [x.date().isocalendar()[1] for x in trend_stock_df.index]

    # so now lets chop into weeks
    trend_stock_df = trend_stock_df.copy()
    trend_stock_df['week_diff'] = trend_stock_df.week.diff()
    trend_stock_df['week_diff'] = [x if x == 0 else 1.0 for x in trend_stock_df['week_diff']]
    trend_stock_df.loc[trend_stock_df.index[0], 'week_diff'] = 0.0

    trend_stock_df['week_number'] = trend_stock_df['week_diff'].cumsum()
    trend_stock_df['week_number'] = trend_stock_df['week_number'].apply(int)

    # print(trend_stock_df.head())
    # so now we can have week summary
    # need to get the week number, the weeks open date, the weeks open, close, high, low, and number of weeks, included
    # then we go and find out the corresponding inputs
    week_summary = trend_stock_df.groupby(['week_number']).agg({
        'week': 'count',
        'High': 'max',
        'Low': 'min',
    })

    week_summary = week_summary[week_summary.week >= 5]
    week_summary['Open_date'] = [trend_stock_df.loc[trend_stock_df.week_number == week_no, :].apply(pd.Series.first_valid_index).Open
                                 for week_no in week_summary.index]
    week_summary['Open'] = [trend_stock_df.loc[date, 'Open'] for date in week_summary.Open_date]

    week_summary['Close_date'] = [trend_stock_df.loc[trend_stock_df.week_number == week_no, :].apply(pd.Series.last_valid_index).Close
                                  for week_no in week_summary.index]
    week_summary['Close'] = [trend_stock_df.loc[date, 'Close'] for date in week_summary.Close_date]



    # let do a rename of the columns
    # now lets form the ys

    # these are the Ys we predict, and we need to scale them
    week_summary['close_open'] = week_summary.Close - week_summary.Open  # can be easily 1 or 0 it is binary
    week_summary['close_open'] = [0 if x > 0 else 1 for x in week_summary['close_open']]
    week_summary['close_low'] = (week_summary.Close - week_summary.Low) / week_summary.Open
    week_summary['high_close'] = (week_summary.High - week_summary.Close) / week_summary.Open
    week_summary['decision'] = ['short' if high_range > low_range else 'long'
                                for high_range, low_range
                                in zip(week_summary.high_close, week_summary.close_low)]  # this one can also be binary

    # if high-close > clow-low: means close has a bigger gap toward high, so we should short, at position high
    # close the position at predicted close price, if cannot close? we set a max loss controller
    # if cannot hit the targeted predicted close price, we will sell at the next week open, or compare to the next weeks
    # decision...
    # let have these numbers first

    # the strategy is that, if max inc > max dec?, then long, else short
    # buy at the beginning, sell at both up/down threshhold
    # or we make use of the close?
    # or we buy in at the price = open - delta(close-low)
    # and set the sell price at open + delta(close - open)

    # now based on the weeks to predict to get all the Xs and Ys
    starting_week_number = week_summary.index[weeks_to_predict - 1]
    week_summary = week_summary[starting_week_number:]

    inputs = []
    targets = []
    week_numbers = []

    for index, row in week_summary.iterrows():

        # now lets prepare
        an_input = trend_stock_df[trend_stock_df.week_number.isin(range(index - weeks_to_predict, index))][trend_columns]
        # make an_input to a list of numbers
        an_input = np.array(an_input).flatten()

        # let's use close minus open first
        a_target = row[predict_what]

        inputs.append(an_input)
        targets.append(a_target)
        week_numbers.append(row.Open_date)

    # let's make some documentation before dive into the machine learing part
    # Ys can be
    # print(week_summary.tail())
    # print(week_summary.head())

    # so i need to make x numpy arrays first
    # then collapse x into a single array
    # and put x together
    # and pass to tts, the clf
    print(f'total number of weeks is {len(week_summary)}')

    return np.stack(tuple(inputs)), targets, week_numbers


def create_trend_files(keywords_list):
    return 0


