
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
    # all_tests = all_tests.groupby(['KEYWORDS', 'MODEL_DESC', 'WEEKS_IN_TRAINS_SIZE', 'TRAIN_SIZE', 'SCALED'])[['SCORE']].mean()
    # all_tests.reset_index(inplace=True)
    # print(tada.head())

    return all_tests


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    all_tests = check_all_records('keywords_combo_with_SPY_experiment.csv')
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

    weeks_in_train_size_summary = all_tests.groupby(['KEYWORDS', 'MODEL_DESC', 'SCALED'])['SCORE'].mean()
    print(weeks_in_train_size_summary)

    # print(all_tests[(all_tests.CASCADE) & (all_tests.BY_DATE_OR_WEEK == 'DAY')].sort_values(by='SCORE'))

    # plots...
    # all_tests = pd.DataFrame()
    control = all_tests[all_tests.KEYWORDS == 'compare_group_True_by_week']
    tf_0 = all_tests[all_tests.MODEL_DESC == 'tf_gen_0']
    tf_0.SCORE.hist(bins=20)

    control = all_tests[all_tests.KEYWORDS == 'compare_group_True_by_week']
    control.SCORE.hist()

    not_control = all_tests[all_tests.KEYWORDS != 'compare_group_True_by_week']
    not_control.SCORE.hist()

    control.SCORE.hist()

    control.SCORE.mean()

    not_control.SCORE.mean()

    top_holding_names = all_tests[all_tests.KEYWORDS != 'top_holding_names_True_by_week']
    top_holding_names.SCORE.hist()

    top_holding_tf = top_holding_names[top_holding_names.MODEL_DESC == 'tf_gen_0']
    top_holding_tf.SCORE.hist()

    control.SCORE.hist()

    top_holding_tf.SCORE.hist()

    top_holding_tf.SCORE.mean()



