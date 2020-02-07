# this is the machine learning block
# where models are trained, saved, and tested
# we will make sure the cpu is 100% running, all the time, and with the right experiment design
# we will find the right pattern


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def train_and_predict(X, y, test_size=0.8):

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

    # random_state=0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    clf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=10)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def train_and_predict_one_by_one(X, y, test_size=0.8):

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

    # random_state=0
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    clf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=10)
    # clf.fit(X_train, y_train)

    correct_ones = 0
    train_size = int(len(y) * test_size)
    test_size = len(y) - train_size
    for i in range(test_size):
        X_train = X[i: i + train_size, :]
        y_train = y[i: i + train_size]
        X_test = X[[i + train_size], :]
        y_test = y[i + train_size]
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        if prediction == y_test:
            correct_ones += 1

    return correct_ones / test_size


