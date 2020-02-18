
from supports import general_path
from datetime import datetime
from datetime import timedelta
import os
import pathlib
import pandas as pd

def check_all_records():
    all_tests = pd.read_csv(os.path.join(general_path(), 'all_tests.csv'), index_col=0)
    all_tests['number_of_kw'] = [len(x.split('_')) for x in all_tests.KEYWORDS]
    return all_tests


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    all_tests = check_all_records()
    print(all_tests.sort_values(by='SCORE'))

    # and now lets print out some summaries
    cascade_summary = all_tests.groupby('CASCADE')['SCORE'].mean()
    print(cascade_summary)

    week_day_summary = all_tests.groupby('BY_DATE_OR_WEEK')['SCORE'].mean()
    print(week_day_summary)

    kw_summary = all_tests.groupby('KEYWORDS')['SCORE'].mean()
    print(kw_summary)

    weeks_in_train_size_summary = all_tests.groupby(['BY_DATE_OR_WEEK', 'WEEKS_IN_TRAINS_SIZE'])['SCORE'].mean()
    print(weeks_in_train_size_summary)

    print(all_tests[(all_tests.CASCADE) & (all_tests.BY_DATE_OR_WEEK == 'DAY')].sort_values(by='SCORE'))

