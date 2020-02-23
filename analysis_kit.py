
from supports import general_path
from datetime import datetime
from datetime import timedelta
import os
import pathlib
import pandas as pd

def check_all_records(file_name='all_tests.csv'):
    all_tests = pd.read_csv(os.path.join(general_path(), file_name), index_col=0)
    # all_tests['number_of_kw'] = [len(x.split('_')) for x in all_tests.KEYWORDS]
    # average based on methods
    tada = all_tests.groupby(['KEYWORDS', 'MODEL_DESC', 'WEEKS_IN_TRAINS_SIZE', 'TRAIN_SIZE', 'SCALED'])[['SCORE']].mean()
    tada.reset_index(inplace=True)
    # print(tada.head())

    return tada


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    all_tests = check_all_records('results_for_bib_trend_independent.csv')
    print(all_tests.sort_values(by='SCORE').tail(30))
    print(all_tests.sort_values(by='SCORE').head(30))

    # and now lets print out some summaries
    # cascade_summary = all_tests.groupby('CASCADE')['SCORE'].mean()
    # print(cascade_summary)

    cascade_summary = all_tests.groupby('TRAIN_SIZE')['SCORE'].mean()
    print(cascade_summary)

    # week_day_summary = all_tests.groupby('BY_DATE_OR_WEEK')['SCORE'].mean()
    # print(week_day_summary)

    kw_summary = all_tests.groupby(['SCALED', 'MODEL_DESC', 'KEYWORDS'])['SCORE'].mean()
    print(kw_summary)

    weeks_in_train_size_summary = all_tests.groupby(['KEYWORDS', 'MODEL_DESC', 'WEEKS_IN_TRAINS_SIZE'])['SCORE'].mean()
    print(weeks_in_train_size_summary)

    # print(all_tests[(all_tests.CASCADE) & (all_tests.BY_DATE_OR_WEEK == 'DAY')].sort_values(by='SCORE'))


