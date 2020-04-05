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
from tensorflow import keras





def models_selection(parameters):
    """

    :param parameters: pandas series...
    :return:
    """

    if parameters is None:
        return None, None

    model_name = parameters['model_name']

    if model_name == 'SVC':
        # now we know it is a SVC, we need kernel, and C
        kernel = parameters['kernel']
        C = parameters['C']

        model = SVC(kernel=kernel, C=C)
        # now lets write the model_description
        model_desc = f'SVC_kernel_{kernel}_C_{C}'

    elif model_name == 'RandomForestClassifier':

        # lets do the similar stuff as above
        max_depth = parameters['max_depth']
        n_estimators = parameters['n_estimators']
        max_features = parameters['max_features']

        model = RandomForestClassifier(max_depth=max_depth,
                                       n_estimators=n_estimators,
                                       max_features=max_features)
        model_desc = f'RandomForestClassifier_max_depth_{max_depth}_n_estimators_{n_estimators}_max_features_{max_features}'

    elif model_name == 'MLPClassifier':

        hidden_layer_sizes = parameters['hidden_layer_sizes']
        max_iter = parameters['max_iter']

        model = MLPClassifier(solver='lbfgs',
                              alpha=1e-4,
                              hidden_layer_sizes=hidden_layer_sizes,
                              random_state=1,
                              max_iter=max_iter)
        model_desc = f'MLPClassifier_hidden_layer_sizes_{hidden_layer_sizes}_max_iter_{max_iter}'

    elif model_name == 'tf_gen_0':

        model = keras.Sequential()

        # add layers one by one here
        # model.add(keras.layers.Embedding(10000, 10))
        # what is the embedding layer.. it transforms the input data array into high dimensional vectors...
        # model.add(keras.layers.GlobalAveragePooling1D())

        # model.add(keras.layers.Dense(5, activation='relu'))    # linear rectify
        model.add(keras.layers.Dense(10, activation='sigmoid'))    # linear rectify
        # model.add(keras.layers.Dense(10, activation='sigmoid'))
        # model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))  # good or bad, so sigmoid, for the 1,0 label

        # check the model summary like this
        # model.summary()

        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

        model_desc = model_name

    elif model_name == 'tf_gen_1':

        model = keras.Sequential()

        # add layers one by one here
        model.add(keras.layers.Embedding(10000, 10))
        # what is the embedding layer.. it transforms the input data array into high dimensional vectors...
        model.add(keras.layers.GlobalAveragePooling1D())

        # model.add(keras.layers.Dense(5, activation='relu'))    # linear rectify
        # model.add(keras.layers.Dense(20, activation='sigmoid'))    # linear rectify
        # model.add(keras.layers.Dense(3, activation='sigmoid'))
        model.add(keras.layers.Dense(10, activation='sigmoid'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))  # good or bad, so sigmoid, for the 1,0 label

        # check the model summary like this
        # model.summary()

        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

        model_desc = model_name

    else:
        return

    # classifier = [
    #               KNeighborsClassifier(3),
    #               SVC(kernel="linear", C=0.025),
    #               SVC(gamma=2, C=1),
    #               GaussianProcessClassifier(1.0 * RBF(1.0)),
    #               DecisionTreeClassifier(max_depth=5),
    #               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #               MLPClassifier(alpha=1, max_iter=1000),
    #               MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 5), random_state=1, max_iter=5000),
    #               AdaBoostClassifier(),
    #               GaussianNB(),
    #               QuadraticDiscriminantAnalysis()]

    return model, model_desc


def split_train_and_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    print(f'train size is {len(X_train)}')
    print(f'test size is {len(X_test)}')
    return X_train, X_test, y_train, y_test


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
    print(f'train size is {len(X_train)}')
    print(f'test size is {len(X_test)}')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=1000)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def train(X, y, test_size=0.8):

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
    print(f'train size is {len(X_train)}')
    print(f'test size is {len(X_test)}')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def train_and_predict_one_by_one(X, y, weeks, test_size=0.8):

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
    clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 5), random_state=1, max_iter=5000)
    # clf.fit(X_train, y_train)

    # lets write some explanation here
    correct_ones = 0
    train_size = int(len(y) * test_size)
    test_size = len(y) - train_size
    print(f'train size is {train_size}')
    print(f'test size is {test_size}')
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


