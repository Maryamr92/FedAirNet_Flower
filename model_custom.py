import tensorflow as tf
from tensorflow.keras import backend as K

from keras.layers import Input, LSTM, Dropout, BatchNormalization, Dense
from flwr.common.parameter import ndarrays_to_parameters
from typing import List, Dict
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

# other configurations
tf.get_logger().setLevel('ERROR')
K.clear_session()
print("Keras backend is functioning correctly.")

class LSTMModel(Model):
    # def __init__(self, input_shape, pre_len: int):
    #     super(LSTMModel, self).__init__()
    #     print('input_shape model: ' + str(input_shape))
    #     self.input_shape_ = input_shape
    #     self.pre_len_ = pre_len
    #     self.lstm1 = LSTM(256, activation='relu', return_sequences=True, input_shape=self.input_shape_)
    #     self.dropout1 = Dropout(0.2)
    #     self.lstm2 = LSTM(256, activation='relu', return_sequences=True)
    #     self.dropout2 = Dropout(0.2)
    #     self.lstm3 = LSTM(128, activation='relu', return_sequences=True)
    #     self.dropout3 = Dropout(0.2)
    #     self.lstm4 = LSTM(128, activation='sigmoid', return_sequences=True)
    #     self.dropout4 = Dropout(0.2)
    #     # self.batch_norm = BatchNormalization()
    #     self.dense1 = Dense(128, activation='relu')
    #     self.dropout5 = Dropout(0.2)
    #     self.output_layer = Dense(self.pre_len_)

    def __init__(self, input_shape, pre_len: int):
        super(LSTMModel, self).__init__()
        print('input_shape model: ' + str(input_shape))
        self.input_shape_ = input_shape
        self.pre_len_ = pre_len
        self.lstm1 = LSTM(128, activation='relu', return_sequences=True, input_shape=self.input_shape_)
        self.dropout1 = Dropout(0.2)
        self.lstm2 = LSTM(128, activation='relu', return_sequences=False)
        self.dropout2 = Dropout(0.2)
        self.output_layer = Dense(self.pre_len_)

    def call(self, inputs):
        x = self.lstm1(inputs)
        # x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        # x = self.dense1(x)
        # x = self.dropout5(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        config = super(LSTMModel, self).get_config()
        config.update({
            "input_shape": self.input_shape_,
            "pre_len": self.pre_len_
        })
        return config


def create_model(model, optimizer, loss_fn, metrics, input_shape):
    """Compile and build the model with the given configuration."""


    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    model.build(input_shape=(None,) + input_shape)
    # model.summary()

def set_layers_trainable(model, layers_to_freeze):
    for layer in model.layers:
        if layer.name in layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True
    return model

# define weighted_mean_absolute_percentage_error and other eveluation metrics
def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
    total_sum = np.sum(Y_true)
    average = []
    for i in range(len(Y_true)):
        for j in range(len(Y_true[1])):
            if Y_true[i][j] > 0:
                temp = (Y_true[i][j] / total_sum) * np.abs((Y_true[i][j] - Y_pred[i][j]) / Y_true[i][j])
                average.append(temp)
    return np.sum(average)


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Custom metric: R-squared (R²)
def r_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - residual / (total + tf.keras.backend.epsilon())
    return r2

def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def main(input_shape, pre_len):
    # Create an instance of the model
    model = LSTMModel(input_shape, pre_len)

    model = set_layers_trainable(model, ['lstm1', 'lstm2'])  # Freeze 'lstm1' and 'lstm2'

    # Define optimizer, loss function, and metrics
    optimizer = tf.keras.optimizers.Adam()
    # loss_fn = tf.keras.losses.MeanAbsoluteError()
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    metrics = ['mse', rmse]

    # Compile and build the model using the function
    create_model(model, optimizer, loss_fn, metrics, input_shape=input_shape)

    return model
