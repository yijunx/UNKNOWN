import numpy as np
import DataModel
import pandas as pd
import math
# reload Model each time this thing is run in order to test
import importlib
import os


import pyTorchModel
import DataModel
import datetime
from supports import kw_dicts
from pull_stock import pull_stock_nicely
from pull_trend import pull_trend_weekly
from supports import general_path
from datetime import datetime

from datetime import timedelta

# importlib.reload(DataModel)
# importlib.reload(pyTorchModel)


# get the files ready first
kw_dict = kw_dicts()

# can create temp kw_dict here...., and merge into the big dic..

def get_ready_info(a_kw_dict, pull_stock=True, pull_trend=True, kw_filter=None):

    stock_names = []
    kws = []
    keys = []

    for key in a_kw_dict:
        if kw_filter is None:
            # store the information
            stock_name = key.split('-')[0]
            stock_names.append(stock_name)
            kws.append(kw_dict[key])
            keys.append(key)

            if pull_trend:
                pull_trend_weekly(kw_small_name=key, number_of_weeks=200, relative_to_each_other=False)
        else:
            if kw_filter in key:
                stock_name = key.split('-')[0]
                stock_names.append(stock_name)
                kws.append(kw_dict[key])
                keys.append(key)

                if pull_trend:
                    pull_trend_weekly(kw_small_name=key, number_of_weeks=200, relative_to_each_other=False)


    # pull stock
    if pull_stock:
        for a_stock_name in list(set(stock_names)):
            # print(f'pulling {a_stock_name}')
            pull_stock_nicely(a_stock_name, number_of_weeks=200)

    return keys, stock_names, kws

keys, stock_names, kws = get_ready_info(kw_dict, False, False, 'bio_stuff_meaning')
# the_great_looper(keys, stock_names, kws)

# use the new experiment kit to the get the great looper
def the_great_looper(key_list, stock_names_list, kws_list):
    start_timing = datetime.now()
    print(f'Start time: {start_timing}')
    # create a number of datamodels first
    # the kw lists

    # create some table to store the data after run
    # and then save it somewhere after run
    result = pd.DataFrame()

    experiment_number = 1

    iterations = 20
    number_of_data_models = len(key_list)

    total_experiment_count = iterations * number_of_data_models

    for key, stock_name, kws in zip(key_list, stock_names_list, kws_list):

        # form the data model
        m = DataModel.DataModel(f'{key}_False_by_week.csv',  # the trend file
                                f'{stock_name}.csv',    # the stock file
                                'close_open',
                                'somefilenamehere')

        # let's loop around the weeks to predict....also, but later
        # loop around number of epochs and learning rate, but later

        m.form_X_y(weeks_to_predict=8, scaled=False, div_100=True, flatten=False)

        for _ in range(iterations):

            print(f'running experiment {experiment_number} / {total_experiment_count}')
            train_loader, val_loader = pyTorchModel.get_train_and_test_data_loader_from_data_model(m, percentage_in_train_set=0.7)

            lr = 0.001
            n_epochs = 600
            kernel_size = 3

            trained_model, train_losses, val_losses = pyTorchModel.train_kit(train_loader, val_loader,
                                                                             number_of_kws=len(m.keywords),
                                                                             kernel_size=kernel_size,
                                                                             pool_size=1,
                                                                             weeks_to_predict=m.weeks_to_predict,
                                                                             lr=lr, n_epochs=n_epochs)

            # the train_kit should return the model and train losses and val losses
            total_accuracy, one_zero_accuracy, zero_one_accuracy = pyTorchModel.eval_kit(val_loader, trained_model)

            print(f'avg train losses is {sum(train_losses) / len(train_losses)}')
            print(f'avg val losses is {sum(val_losses) / len(val_losses)}')

            # document the result
            # result_cols = ['key', 'kws', 'weeks_to_predict', 'train_loss', 'val_loss', 'lr', 'n_epochs',
            #            'total_accuracy', 'one_zero_accuracy', 'zero_one_accuracy'],
            result.loc[experiment_number, 'key'] = key
            result.loc[experiment_number, 'kws'] = kws
            result.loc[experiment_number, 'weeks_to_predict'] = m.weeks_to_predict
            result.loc[experiment_number, 'train_loss'] = float(sum(train_losses) / len(train_losses))
            result.loc[experiment_number, 'val_loss'] = float(sum(val_losses) / len(val_losses))
            result.loc[experiment_number, 'lr'] = lr
            result.loc[experiment_number, 'n_epochs'] = n_epochs
            result.loc[experiment_number, 'total_accuracy'] = total_accuracy
            result.loc[experiment_number, 'one_zero_accuracy'] = one_zero_accuracy
            result.loc[experiment_number, 'zero_one_accuracy'] = zero_one_accuracy
            result.loc[experiment_number, 'kernel_size'] = kernel_size
            # result.loc[experiment_number, 'key'] = key
            # result.loc[experiment_number, 'key'] = key
            experiment_number += 1

    result.to_csv(os.path.join(general_path(), 'result7_only_bio_meaningful.csv'))
    end_timing = datetime.now()
    print(f'time taken: {end_timing - start_timing}')
    return result

the_great_looper(keys, stock_names, kws)
