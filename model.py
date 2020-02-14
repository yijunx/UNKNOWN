import pandas as pd
import os
from preparation_kit import form_X_y_from_daily_data
from preparation_kit import form_X_y_from_weekly_data
from warp_drive import train_and_predict_one_by_one
from warp_drive import train_and_predict


class Model:
    """
    This model keeps track of the trend data and stock data, it can read files, generate tests and record
    and predict
    """
    def __init__(self, keywords_file_link, stock_file_link):
        self.keywords_file_link = keywords_file_link
        self.stock_file_link = stock_file_link
        self.start_date = 0
        self.end_date = 0
        self.X = 0
        self.y = 0
        # check if this is by week or by day
        # thus i need to change the pulling section
        self.trend_by_day = True
        if 'by_week' in keywords_file_link:
            self.trend_by_day = False

    def from_x_y(self):
        if self.trend_by_day:
            self.X, self.y, weeks = form_X_y_from_daily_data(self.keywords_file_link,
                                                             self.stock_file_link,
                                                             weeks_to_predict=4)
        else:
            self.X, self.y, weeks = form_X_y_from_weekly_data(self.keywords_file_link,
                                                              self.stock_file_link,
                                                              weeks_to_predict=4)
        self.start_date = weeks[0]
        self.end_date = weeks[-1]


    def train_and_predict(self, model_selection=0):



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
