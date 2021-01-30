import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers import Dense, BatchNormalization, Conv1D, Conv2D, Conv3D, Flatten, Dropout, MaxPooling3D, MaxPooling2D, MaxPooling1D
from keras import Model, Input
from keras import backend as K
from tensorflow.keras import regularizers
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
from plotting import plot_loss, plot_scatter
from utils import SlidingWindow

parser = argparse.ArgumentParser(description='Args for the CNN model')
parser.add_argument('model_name', type=str, help='Name of the model, for better management')
parser.add_argument('epoch_num', type=int, help='Number of training epochs')
parser.add_argument('learning_rate', type=float, help='Learning rate of the model')
parser.add_argument('data', type=str, help='Use small data or large data')
args = parser.parse_args()
# model name as command line input
model_name = args.model_name
# epoch num as input
epoch_num = args.epoch_num
# learning rate as input
lr = args.learning_rate
# small or large data
data_size = args.data

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'
DATA_PATH = ''
IMG_PATH = cwd + 'plots'
MODEL_PATH = cwd + 'models/'


def load_data(x_path, y_path):
    x = np.load(x_path, mmap_mode='r+')
    y = np.load(y_path, mmap_mode='r+')
    # x = np.load(x_path)
    # y = np.load(y_path)
    print(x.dtype, y.dtype)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def train_model(drop_ps, lr, x_train, y_train, x_val, y_val, epochs, batch_size):
    p1, p2, p3, p4, p5 = drop_ps
    seq_len = 240
    h, w = 36, 64

    data_input = tf.keras.layers.Input(shape=(seq_len, h, w, 3))
    # spatial temporal filter
    h = Conv3D(8, (6, 3, 6), strides=(1, 1, 1), padding='SAME', activation='tanh')(data_input)
    h = MaxPooling3D(pool_size=(4, 2, 2))(h)
    h = Dropout(p1)(h)

    h = tf.reshape(h, (-1, 18, 32, 8))

    # spatial filter
    h = Conv2D(16, (3, 6), strides=(1, 1), padding='SAME', activation='tanh')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Dropout(p2)(h)

    h = Conv2D(16, (3, 6), strides=(1, 1), padding='SAME', activation='tanh')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Dropout(p3)(h)

    # linearly combine each channel (neuron); in_size = (bx60, 4, 8, 16), out_size = (bx60, 16)
    h = K.mean(h, axis=(1, 2))
    h = tf.reshape(h, (-1, 60, 16))
    h = Flatten()(h)
    x_pred = Dense(1, activation='tanh', name='x_pred')(h)
    y_pred = Dense(1, activation='tanh', name='y_pred')(h)

    opt = keras.optimizers.Adam(learning_rate=lr)
    model = keras.Model(inputs=data_input, outputs=[x_pred, y_pred])
    model.compile(loss="mean_squared_error", optimizer=opt)

    history = model.fit(x_train, [y_train[:,0], y_train[:,1]], 
                        validation_data=(x_val, [y_val[:,0], y_val[:,1]]), 
                        epochs=epochs, 
                        batch_size=batch_size,)
    return model, history

def main():
    # train the model and save the results
    if data_size == 'small':
        train, val = load_data(DATA_PATH+'x_small.npy', DATA_PATH+'y_small.npy')
    else:
        train, val = load_data(DATA_PATH+'x_all.npy', DATA_PATH+'y_all.npy')

    x_train, y_train = train[0], train[1]
    x_val, y_val = val[0], val[1]

    # hyperparameters
    drop_ps = [0.2, 0.2, 0.2, 0, 0]
    epochs = epoch_num
    batch_size = 20
    print('Training {} for {} epochs'.format(model_name, epoch_num))
    # the model
    model, history = train_model(drop_ps, lr, x_train, y_train, x_val, y_val, epochs, batch_size)
    print(model.summary())
    # path for saving images
    os.mkdir(os.path.join(IMG_PATH, model_name))
    # plot the losses and velocity vectors
    plot_loss(history, epochs, model_name)
    plot_scatter(model, x_train, y_train, x_val, y_val, model_name)

    # # save the predictions
    # y_pred_train = model.predict(x_train)
    # np.save('y_pred_all.npy', y_pred_train)

    # save the model
    model.save(MODEL_PATH+model_name)

if __name__ == '__main__':
    main()