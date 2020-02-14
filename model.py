import pandas as pd
import os
from preparation_kit import form_X_y_from_daily_data
from preparation_kit import form_X_y_from_weekly_data
from warp_drive import train_and_predict_one_by_one
from warp_drive import train_and_predict
from warp_drive import models_selection
from warp_drive import split_train_and_test
from supports import general_path


class Model:
    """
    This model keeps track of the trend data and stock data, it can read files, generate tests and record
    and predict
    """
    def __init__(self, keywords_file_name, stock_file_name, model_name, parameters):
        self.keywords_file_name = keywords_file_name
        self.stock_file_name = stock_file_name
        self.start_date = 0
        self.end_date = 0
        self.training_size = 0
        self.score = 0
        self.X = []
        self.y = []
        self.time_stamps = []
        self.weeks_to_predict = 0
        self.keywords = self.keywords_file_name.split('_by')[0].split('_')
        self.stock_name = self.stock_name.split('_end_at_')[0]
        self.model, self.model_desc = models_selection(model_name, parameters)
        # check if this is by week or by day
        # thus i need to change the pulling section

        if 'by_week' in keywords_file_name:
            self.trend_by_day = False
        else:
            self.trend_by_day = True

    def __repr__(self):
        print(self.model_desc)
        print(self.keywords)
        print(self.stock_name)
        return f'{self.model_desc}'

    def from_x_y(self, weeks_to_predict):
        if self.trend_by_day:
            self.X, self.y, self.time_stamps = form_X_y_from_daily_data(self.keywords_file_name,
                                                                        self.stock_file_name,
                                                                        weeks_to_predict=weeks_to_predict)
        else:
            self.X, self.y, self.time_stamps = form_X_y_from_weekly_data(self.keywords_file_name,
                                                                         self.stock_file_name,
                                                                         weeks_to_predict=weeks_to_predict)
        self.start_date = self.time_stamps[0]
        self.end_date = self.time_stamps[-1]
        self.weeks_to_predict = weeks_to_predict

    def reselect_model(self, model_name, parameters):
        self.model, self.model_desc = models_selection(model_name, parameters)

    def fit_and_predict_normal(self, test_size):
        X_train, X_test, y_train, y_test = split_train_and_test(self.X, self.y, test_size)
        self.model.fit(X_train, y_train)
        self.training_size = len(X_train)
        self.score = self.model.score(X_test, y_test)
        print(f'train size is {len(X_train)}')
        print(f'test size is {len(X_test)}')

        # print some result, or picture
        y_predict = self.model.predict(X_test)

        # result df
        result_df = pd.DataFrame(index=range(len(y_test)))
        result_df['y_test'] = y_test
        result_df['y_predict'] = y_predict
        print(result_df)

        # at the end we must log it, log the result here
        self.log(cascade=False)

    def fit_and_predict_cascade(self, test_size):

        # at the end we must log it
        # lets write some explanation here
        correct_ones = 0
        train_amount = int(len(self.y) * test_size)
        self.training_size = train_amount
        test_amount = len(self.y) - train_amount
        print(f'train size is {train_amount}')
        print(f'test size is {test_amount}')
        result_df = pd.DataFrame(index=range(test_amount), columns=['y_test', 'y_predict'])
        for i in range(test_amount):
            X_train = self.X[i: i + train_amount, :]
            y_train = self.y[i: i + train_amount]
            X_test = self.X[[i + train_amount], :]
            y_test = self.y[i + train_amount]
            self.model.fit(X_train, y_train)
            prediction = self.model.predict(X_test)

            # log into result
            result_df.loc[i, 'y_test'] = y_test
            result_df.loc[i, 'y_predict'] = prediction

            if prediction == y_test:
                correct_ones += 1

        # print some result, or picture
        # need to calculate the score here
        print(result_df)

        self.score = correct_ones / test_amount
        # at the end we must log it
        self.log(cascade=True)

    def log(self, cascade):

        # fetch the file
        all_tests = pd.read_csv(os.path.join(general_path(), 'all_tests.csv'), index_col=0)
        # write data into the file
        row_index = len(all_tests)
        all_tests.loc[row_index, 'KEYWORDS'] = '_'.join(self.keywords)
        all_tests.loc[row_index, 'STOCK_NAME'] = self.stock_name
        all_tests.loc[row_index, 'MODEL_DESC'] = self.model_desc
        all_tests.loc[row_index, 'TRAIN_SIZE'] = self.training_size
        all_tests.loc[row_index, 'WEEKS_IN_TRAINS_SIZE'] = self.weeks_to_predict
        all_tests.loc[row_index, 'SCORE'] = self.score
        all_tests.loc[row_index, 'TREND_START_DATE'] = self.start_date
        all_tests.loc[row_index, 'TREND_END_DATE'] = self.end_date
        all_tests.loc[row_index, 'BY_DATE_OR_WEEK'] = 'DAY' if self.trend_by_day else 'WEEK'
        all_tests.loc[row_index, 'CASCADE'] = cascade
        # all_tests.loc[row_index, 'KEYWORDS'] = '_'.join(self.keywords)

        # output the results
        all_tests.to_csv(os.path.join(general_path(), 'all_tests.csv'))

    def execute(self):
        return

    def score(self):
        return 0

    def predict(self):
        return 0

    def record_results(self):
        return 0

    def compare_with_another_model(self):
        return 0

    def analyze_trend(self, trend_saved):

        # this function cross the changes in the today pull and old pull
        # see if therer is any discrepancies
        # those unexplanable stuff....

        # this is check if the trend pulled is consistent
        return 0

    def pull_and_compare(self):

        return 0
