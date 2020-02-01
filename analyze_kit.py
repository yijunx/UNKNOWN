import pandas as pd
from supports import general_path
import os


def read_trend(trend_file_name):
    # pulled_at_2020-01-30_end_at_2020-1-11_for_35_weeks.csv
    return pd.read_csv(os.path.join(general_path(), trend_file_name))


def read_stock(stock_file_name):

    return 0


def first_pass_analysis(trend_df, stock_df):
    return 0


