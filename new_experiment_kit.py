import numpy as np
import DataModel
import pandas as pd
import math
# reload Model each time this thing is run in order to test
import importlib


import pyTorchModel
import DataModel
import datetime
from supports import kw_dicts
from supports import pull_stock_nicely
from supports import pull_trend_weekly
from datetime import datetime
from datetime import timedelta

importlib.reload(DataModel)
importlib.reload(pyTorchModel)


# get the files ready first
kw_dict = kw_dicts()



def get_ready_info(a_kw_dict, pull_stock=True, pull_trend=True):

    stock_names = []
    kws = []
    keys = []

    for key in a_kw_dict:
        # store the information
        stock_name = key.split('-')[0]
        stock_names.append(stock_name)
        kws.append(kw_dict[key])
        keys.append(key)

        if pull_trend:
            for a_keyword in kw_dict[key].split('_'):
                pull_trend_weekly(a_keyword, number_of_weeks=200)

    # pull stock
    if pull_stock:
        for a_stock_name in list(set(stock_names)):
            pull_stock_nicely(a_stock_name, number_of_weeks=200)

    return keys, stock_names, kws

# use the new experiment kit to the get the great looper
def the_great_looper(key_list, stock_names_list, kws_list):


    start_timing = datetime.now()
    print(f'Start time: {start_timing}')
    # create a number of datamodels first
    # the kw lists

    # create some table to store the data after run
    # and then save it somewhere after run
    result_cols = ['key', 'kws', 'train_loss', 'val_loss', 'weeks_to_predict', '', '']
    result = pd.DataFrame(columns=result_cols)

    for key, stock_name, kws in zip(key_list, stock_names_list, kws_list):

        # form the data model
        m = DataModel.DataModel(f'{key}_False_by_week.csv',  # the trend file
                                f'{stock_name}.csv',    # the stock file
                                'close_open',
                                'somefilenamehere')

        # let's loop around the weeks to predict....also, but later
        # loop around number of epochs and learning rate, but later

        m.form_X_y(weeks_to_predict=30, scaled=False, div_100=True, flatten=False)

        train_loader, val_loader = pyTorchModel.get_train_and_test_data_loader_from_data_model(m)

        trained_model, train_losses, val_losses = pyTorchModel.train_kit(train_loader, val_loader,
                                                                         number_of_kws=len(m.keywords),
                                                                         kernel_size=6,
                                                                         pool_size=1,
                                                                         weeks_to_predict=m.weeks_to_predict,
                                                                         lr=0.001, n_epochs=400)

        # the train_kit should return the model and train losses and val losses
        pyTorchModel.eval_kit(val_loader, trained_model)

        print(f'avg train losses is {sum(train_losses) / len(train_losses)}')
        print(f'avg val losses is {sum(val_losses) / len(val_losses)}')







# then let it run over night...

# now we need to make multiple m and multiple weeks to predict

# and document the train loss and val loss....





