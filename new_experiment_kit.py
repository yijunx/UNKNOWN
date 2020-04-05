import numpy as np
import DataModel
import pandas as pd
import math
# reload Model each time this thing is run in order to test
import importlib
importlib.reload(DataModel)

# before run, pull a bunch of data first


# create a number of datamodels first


# then let it run over night...


m = DataModel.DataModel('economics2_False_by_week.csv',  # the trend file
          'long_SPY',    # the stock file
          # tf_gen_0,      # the model...
          'close_open',
          'somefilenamehere')

# now it is the time for the
m.form_X_y(weeks_to_predict=30, scaled=False, div_100=True, flatten=False)



