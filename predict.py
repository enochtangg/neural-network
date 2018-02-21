from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import argmax
from keras.utils import to_categorical
from matplotlib import pyplot

import numpy
import keras
import pandas
import collections


_training_split = 0.8


def split_to_training(dataframe):
    train_df = dataframe[:int(len(dataframe) * _training_split)]

    return train_df


def split_to_testing(dataframe):
    test_df = dataframe[int(len(dataframe) * _training_split):]

    return test_df


def split_to_x(dataframe):
    x = dataframe[features].values

    return x


def split_to_y(dataframe):
    y = dataframe[labels].values

    return y


def encode_dataframe(label_y):
    encoder = LabelEncoder()
    encoder.fit(label_y)
    encoded_Y = encoder.transform(label_y)
    new_y = np_utils.to_categorical(encoded_Y)

    return new_y


class KerasNeuralNetwork:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        _number_of_test_models = 2

        base_number_layers = int(input_dim ** .5)
        #       group_models { loss_value: respective model {}
        group_models = {}

        for i in range(_number_of_test_models):
            model = Sequential()

            # Base number of layers
            model.add(Dense(input_dim, input_dim=input_dim, activation='sigmoid'))

            # Loop adds an extra hidden layer after each individual model is trained
            for i in range(base_number_layers):
                model.add(Dense(4, activation='sigmoid'))
            model.add(Dense(3, activation='sigmoid'))
            model.summary()

            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
            history = model.fit(self.x_train, self.y_train, epochs=150, validation_split=0.2)
            scores = model.evaluate(self.x_test, self.y_test)

            # Map each loss with its respective model
            # Note that group_models[0] = 'model object' and group_models[1] = the'history object'
            group_models[scores[0]] = model, history

            pyplot.plot(history.history['loss'])
            pyplot.plot(history.history['val_loss'])
            pyplot.title('Loss After Each Epoch')
            pyplot.ylabel('loss')
            pyplot.xlabel('epoch')

            print("{}: {}".format(model.metrics_names[0], scores[0]))
            print("{}: {}%".format(model.metrics_names[1], scores[1] * 100))
            base_number_layers += 1

        pyplot.legend(['2HL', 'val_2HL', '3HL', 'val_3HL'], loc='upper right')
        pyplot.show()

        # sort the dictionary
        ordered_group_models = collections.OrderedDict(sorted(group_models.items()))

        # Use models starting from the lowest loss. If it is overfitted, take the second lowest loss and so on.

        overfitted = False
        for key, value in ordered_group_models.items():
            list_of_loss = value[1].history['loss']
            list_of_val_loss = value[1].history['val_loss']

            for i in range(len(list_of_loss)):
                if list_of_loss[i] < list_of_val_loss[i]:
                    overfitted = True
                    break  # the model is overfitted

            if not overfitted:
                optimized_model = value[0]
                optimized_history = value[1]
                break  # exit loop because we have found optimized model

            overfitted = False

        if not optimized_model:
            optimized_model = ordered_group_models.keys()[0]

        # Evaluating the optimized model
        evaluation = optimized_model.evaluate(self.x_test, self.y_test)
        print("{}: {}".format(optimized_model.metrics_names[0], evaluation[0]))
        print("{}: {}%".format(optimized_model.metrics_names[1], evaluation[1] * 100))

        pyplot.plot(optimized_history.history['loss'])
        pyplot.plot(optimized_history.history['val_loss'])
        pyplot.title('Loss and Val_Loss vs Epoch')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['Loss', 'Val_Loss'], loc='upper right')
        pyplot.show()

dataframe = pandas.read_csv('iris.csv')
df = shuffle(dataframe)
features = list(df.columns.values)[:-1]
labels = list(df.columns.values)[-1]
input_dim = len(list(df.columns.values)[:-1])

train_df = split_to_training(df)
x_train = split_to_x(train_df)
y_train = encode_dataframe(split_to_y(train_df))

test_df = split_to_testing(df)
x_test = split_to_x(test_df)
y_test = encode_dataframe(split_to_y(test_df))

NN = KerasNeuralNetwork(x_train, y_train, x_test, y_test)
NN.run()

