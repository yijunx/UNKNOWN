import pandas as pd
import os
from preparation_kit import form_X_y_from_daily_data
from preparation_kit import form_X_y_from_weekly_data
from warp_drive import models_selection
from warp_drive import split_train_and_test
from supports import general_path
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Model:
    """
    This model keeps track of the trend data and stock data, it can read files, generate tests and record
    and predict
    """
    def __init__(self, keywords_file_name, stock_file_name, model_parameters, predict_what):
        """

        :param keywords_file_name:
        :param stock_file_name:
        :param model_name:
        :param model_parameters:
        """

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
        self.stock_name = self.stock_file_name.split('_end_at_')[0]
        self.model, self.model_desc = models_selection(model_parameters)
        self.predict_what = predict_what
        self.scaled = False
        # check if this is by week or by day
        # thus i need to change the pulling section

        if 'by_week' in keywords_file_name:
            self.trend_by_day = False
        else:
            self.trend_by_day = True

    def __repr__(self):
        return f'{self.model_desc}'

    def form_X_y(self, weeks_to_predict, scaled=False, div_100=True):
        if self.trend_by_day:
            self.X, self.y, self.time_stamps = form_X_y_from_daily_data(self.keywords_file_name,
                                                                        self.stock_file_name,
                                                                        weeks_to_predict=weeks_to_predict,
                                                                        predict_what=self.predict_what,
                                                                        scaled=scaled,
                                                                        div_100=div_100)
            self.start_date = self.time_stamps[0]
            self.end_date = self.time_stamps[-1]
        else:
            self.X, self.y, self.time_stamps = form_X_y_from_weekly_data(self.keywords_file_name,
                                                                         self.stock_file_name,
                                                                         weeks_to_predict=weeks_to_predict,
                                                                         predict_what=self.predict_what,
                                                                         scaled=scaled,
                                                                         div_100=div_100)
            self.start_date = self.time_stamps[0].Open_date
            self.end_date = self.time_stamps[-1].Close_date

        self.weeks_to_predict = weeks_to_predict
        self.scaled = scaled

    def reselect_model(self, parameters):
        self.model, self.model_desc = models_selection(parameters)

    def fit_and_predict_normal(self, test_size, log=True):  # instead of giving test_size, we should give training amount!!!
        X_train, X_test, y_train, y_test = split_train_and_test(self.X, self.y, test_size)
        self.model.fit(X_train, np.array(y_train))
        self.training_size = len(X_train)
        self.score = self.model.score(X_test, y_test)

        # print some result, or picture
        y_predict = self.model.predict(X_test)

        # result df
        result_df = pd.DataFrame(index=range(len(y_test)))
        result_df['y_test'] = y_test
        result_df['y_predict'] = y_predict
        result_df['correct'] = [True if y_t == y_p else False
                                for y_t, y_p in zip(result_df.y_test, result_df.y_predict)]
        print(result_df)

        # print(f'\nscore is {self.score}')
        print(f'correct portion is {len(result_df[result_df.correct]) / len(X_test)}')

        # at the end we must log it, log the result here
        if log:
            self.log(cascade=False)

    def fit_and_predict_cascade(self, test_size, log=True):

        # at the end we must log it
        # lets write some explanation here
        correct_ones = 0
        test_amount = int(len(self.y) * test_size)
        train_amount = len(self.y) - test_amount
        self.training_size = train_amount
        print(f'train size is {train_amount}')
        print(f'test size is {test_amount}')
        result_df = pd.DataFrame(index=range(test_amount), columns=['y_test', 'y_predict'])
        for i in range(test_amount):
            X_train = self.X[i: i + train_amount, :]
            y_train = self.y[i: i + train_amount]
            X_test = self.X[[i + train_amount], :]
            y_test = self.y[i + train_amount]

            # print(X_train.shape, np.array(y_train).shape)
            # print(type(X_train))

            # fitModel =
            self.model.fit(X_train, np.array(y_train))

            prediction = self.model.predict(X_test)[0][0]

            # log into result
            result_df.loc[i, 'y_test'] = y_test
            result_df.loc[i, 'y_predict'] = prediction

            if prediction == y_test:
                correct_ones += 1

        # print some result, or picture
        # need to calculate the score here
        result_df['correct'] = [True if abs(y_t - y_p) <= 0.5 else False
                                for y_t, y_p in zip(result_df.y_test, result_df.y_predict)]
        print(result_df)

        self.score = correct_ones / test_amount
        print(f'correct portion is {len(result_df[result_df.correct]) / test_amount}')

        # at the end we must log it
        if log:
            self.log(cascade=True)

    def log(self, cascade):
        print('log starts!')
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
        all_tests.loc[row_index, 'STUDY_ON_WHAT'] = self.predict_what
        all_tests.loc[row_index, 'SCALED'] = self.scaled

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


if __name__ == "__main__":
    # file names we have
    # 'AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY_by_week.csv'
    # 'biotech_bioinformatics_biotechnology jobs_bioengineering_by_day.csv'
    # 'biotech_bioinformatics_biotechnology jobs_bioengineering_AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY_by_day.csv'
    # 'biotechnology_bioinformatics_biotechnology jobs_bioengineering_virus_health care_by_week.csv'
    # 'biotechnology_bioinformatics_biotechnology jobs_bioengineering_virus_health care_by_day.csv'
    # here comes the test...


    para_random_forest = pd.Series(index=['model_name', 'max_depth', 'n_estimators', 'max_features'],
                                   data=['RandomForestClassifier', 5, 10, 1])

    para_MLP = pd.Series(index=['model_name', 'hidden_layer_sizes', 'max_iter'],
                         data=['MLPClassifier', (20, 3), 2000])

    tf_gen_0 = pd.Series(index=['model_name'],
                         data=['tf_gen_0'])

    m = Model('AMGN_VRTX_BIIB_GILD_REGN_ILMN_ALXN_SGEN_INCY_by_week.csv',
              'BIB_end_at_2020-02-17_for_100_weeks.csv',
              tf_gen_0,
              'close_open')

    m.form_X_y(weeks_to_predict=9, scaled=False, div_100=False)
    m.fit_and_predict_cascade(test_size=0.5, log=False)
