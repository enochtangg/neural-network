# The following code is a classification neural network

# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import argmax
from keras.utils import to_categorical

import numpy
import keras
import pandas

_training_split = 0.8


def split_to_training(dataframe):
    train_df = dataframe[:int(len(dataframe) * _training_split)]

    return train_df


def split_to_testing(dataframe):
    test_df = dataframe[int(len(dataframe) * _training_split):]

    return test_df


def split_to_x(dataframe):
    x = dataframe[list(dataframe.columns.values)[:-1]].values

    return x


def split_to_y(dataframe):
    y = dataframe[list(dataframe.columns.values)[-1]].values

    return y


def encode_dataframe(label_y):
    encoder = LabelEncoder()
    encoder.fit(label_y)
    encoded_Y = encoder.transform(label_y)
    new_y = np_utils.to_categorical(encoded_Y)

    return new_y


def prep_data(filename):
    dataframe = pandas.read_csv(filename)
    df = shuffle(dataframe)

    return df


class KerasNeuralNetwork:

    def __init__(self, x_train, y_train, x_test, y_test, df):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_dim = len(list(df.columns.values)[:-1])

    def run(self):
        _number_of_test_models = 5

        base_number_layers = int(self.input_dim ** .5)
        list_of_loss = []
        list_of_models = []
        list_of_history = []
        for i in range(_number_of_test_models):
            model = Sequential()

            # Base number of layers
            model.add(Dense(self.input_dim, input_dim=self.input_dim, activation='sigmoid'))
            for i in range(base_number_layers):
                model.add(Dense(5, activation='sigmoid'))
            model.add(Dense(3, activation='sigmoid'))
            model.summary()

            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
            list_of_models.append(model)

            history = model.fit(self.x_train, self.y_train, epochs=30)
            list_of_history.append(history)

            # Compare loss after each epoch
            loss_after_each_epoch = history.history['loss']
            avg_roc = abs((loss_after_each_epoch[0] - loss_after_each_epoch[-1]) / len(loss_after_each_epoch))
            list_of_loss.append(avg_roc)
            base_number_layers += 1

            scores = model.evaluate(self.x_test, self.y_test)
            print("{}: {}".format(model.metrics_names[0], scores[0]))
            print("{}: {}%".format(model.metrics_names[1], scores[1] * 100))

        # Take the first trough in list and use that model
        minimal_loss_position = 0;
        for index, value in enumerate(list_of_loss):
            if value < list_of_loss[index + 1]:
                minimal_loss_position = index
                break
        print(list_of_loss)
        print(list_of_models)
        print('The optimal model is at index: {}'.format(minimal_loss_position))
        optimized_model = list_of_models[minimal_loss_position]
        optimized_history = list_of_history[minimal_loss_position]

        # Evaluating the model
        print(optimized_history.history)
        scores = optimized_model.evaluate(self.x_test, self.y_test)
        print("{}: {}".format(optimized_model.metrics_names[0], scores[0]))
        print("{}: {}%".format(optimized_model.metrics_names[1], scores[1] * 100))
