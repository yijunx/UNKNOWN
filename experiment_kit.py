# now let's think about the trade strategy
# then design the experiment...
# regression on high , low
# clf on close - open
# if close > open, buy at low, sell at close/high/open price, and must sell at xx to stop loss?
# if open > close, buy at high, sell at close/low/open price, and must sell at xx to step loss?
# this module will create a lot of models and run them
# to generate a lot of data
# then the analysis kit comes to analyze them to find out the best model to fit different traits

from model import Model
import pandas as pd
from pull_stock import pull_stock
from pull_trend import pull_keywords_trend
from supports import general_path
from datetime import timedelta
from datetime import datetime

# shall let the laptop cpu running at all times...

# experiment on the close_high


# para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
#                                data=['RandomForestClassifier', 5, 10, 1])
#
# para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
#                                data=['RandomForestClassifier', 5, 10, 1])
#
# para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
#                                data=['RandomForestClassifier', 5, 10, 1])
#
# para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
#                                data=['RandomForestClassifier', 5, 10, 1])
#
# para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
#                                data=['RandomForestClassifier', 5, 10, 1])
#
# para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
#                                data=['RandomForestClassifier', 5, 10, 1])
#
# para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
#                      data=['MLPClassifier', (6,), 2000])
# para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
#                      data=['MLPClassifier', (6,), 2000])
# para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
#                      data=['MLPClassifier', (6,), 2000])
# para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
#                      data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                      data=['MLPClassifier', (6,), 2000])
para_MLP2 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                      data=['MLPClassifier', (6, 6), 2000])
para_MLP3 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                      data=['MLPClassifier', (10, 5), 2000])


# lets give kw group name
# so that when it is logged it is easy to understand
# today need to make this experiment kit work - -!

ke_dict = {'top_holding_names': 'AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY',
           'related_words': 'biotechnology_bioinformatics_biotechnology jobs_bioengineering_investment fund_society_economy_biotechnology innovation organization',
           'compare_group': 'apple_pear_root bear',
           'holding+related_words': 'AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY_biotechnology_bioinformatics_biotechnology jobs_bioengineering_investment fund_society_economy_biotechnology innovation organization',
           # '': [],
           # '': [],
           # '': [],
           # '': [],
           # '': [],
           # '': [],
           # '': [],
           # '': [],
           # '': [],
           }

kw_dict = {'debt': 'debt',
           'compare_group': 'apple_pear_root bear',
           'debt_and_others': 'default_derivatives_debt_credit_crisis_gold price',
           'economics': 'inflation_housing_investment_travel_unemployment'}
tf_gen_0 = pd.Series(index=['model_name'],
                     data=['tf_gen_0'])

tf_gen_1 = pd.Series(index=['model_name'],
                     data=['tf_gen_1'])

training_weeks = [1, 3, 5, 7, 9, 11]
training_test_portions = [0.5, 0.6, 0.7, 0.8]
paras = [para_MLP1, para_MLP2, para_MLP3, tf_gen_0, tf_gen_1]



by_day = False
relative_to_each_other = False

if by_day:
    number_of_weeks = 35
else:
    number_of_weeks = 100

end_date = datetime.now().date()
start_date = end_date - timedelta(days=number_of_weeks * 7)

# conversion both to string so that we can pass to the pytrend.. (date to string.. string to date)
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

# form the time_frame
time_frame = f'{start_date} {end_date}'

for item in kw_dict:
    print(f'pulling kw: {item}')
    pull_keywords_trend(kw_dict[item].split('_')[:5], item, time_frame=time_frame,
                        save_folder=general_path(), relative_to_each_other=relative_to_each_other)
# pull stock

# do the experiment
for item in kw_dict:
    for para in paras:
        for a_test_size in training_test_portions:
            for weeks_to_train in training_weeks:

                m = Model(f'{item}_{relative_to_each_other}_{"by_day" if by_day else "by_week"}.csv',
                          'SPY_end_at_2020-2-23_for_100_weeks.csv',
                          para,
                          'close_open',
                          'empty.csv')

                m.form_X_y(weeks_to_predict=weeks_to_train, scaled=False, div_100=False)
                # m.fit_and_predict_cascade(test_size=a_test_size, log=True)
                m.fit_and_predict_normal(test_size=a_test_size, log=True)
                m.fit_and_predict_normal(test_size=a_test_size, log=True)
                m.fit_and_predict_normal(test_size=a_test_size, log=True)

                m.form_X_y(weeks_to_predict=weeks_to_train, scaled=True, div_100=True)
                # m.fit_and_predict_cascade(test_size=a_test_size, log=True)
                m.fit_and_predict_normal(test_size=a_test_size, log=True)
                m.fit_and_predict_normal(test_size=a_test_size, log=True)
                m.fit_and_predict_normal(test_size=a_test_size, log=True)
