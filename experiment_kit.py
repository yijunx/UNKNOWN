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

# shall let the laptop cpu running at all times...

# experiment on the close_high


para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                               data=['RandomForestClassifier', 5, 10, 1])

para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                               data=['RandomForestClassifier', 5, 10, 1])

para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                               data=['RandomForestClassifier', 5, 10, 1])

para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                               data=['RandomForestClassifier', 5, 10, 1])

para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                               data=['RandomForestClassifier', 5, 10, 1])

para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                               data=['RandomForestClassifier', 5, 10, 1])

para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])
para_MLP1 = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                     data=['MLPClassifier', (6,), 2000])

clfs = []

kwfiles = ['biotechnology_bioinformatics_biotechnology jobs_bioengineering_investment fund_society_economy_biotechnology innovation organization_by_day.csv',
           'biotechnology_bioinformatics_biotechnology jobs_bioengineering_investment fund_society_economy_biotechnology innovation organization_by_week.csv',
           'biotechnology_bioinformatics_biotechnology jobs_bioengineering_virus_health care_by_day.csv',
           'biotech_bioinformatics_biotechnology jobs_bioengineering_AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY_by_day.csv',
           'AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY_by_week',
           'apple, pear, root bear',
           '',
           '',
           '']


training_weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
training_test_portion = [0.5, 0.6, 0.7, 0.8]



# and let's form models and run a lot of them
# starting by looping like crazy.. see how many lines of data can be created overnight
