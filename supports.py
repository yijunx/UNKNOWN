"""

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
import pandas as pd
import os


def general_path():
    return r'/Users/yijunxu/Dropbox/UNKNOWN_RESULTS'


def read_results():
    results = pd.read_csv(os.path.join(general_path(), 'all_tests.csv'))
    # do all the necessary stuff in here

    return results

