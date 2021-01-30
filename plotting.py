import numpy as np
import matplotlib.pyplot as plt
import math
import os
import tensorflow as tf
from keras import Model, Input
import keras

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'
DATA_PATH = ''
IMG_PATH = cwd + 'plots'
MODEL_PATH = cwd + 'models/'

# plot the distribution of the x-y coordinates of the target variable
# file a .npy input file, title the tile of the plot
def plot_target_distribution(file, title, save_name):
    y = np.load(file)
    print(y.shape)
    plt.figure()
    plt.scatter(y[:,0], y[:,1], alpha=0.2)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title(title)
    plt.savefig(save_name, dpi=200)

# plot the distribution of angles
def plot_ang_distribution(file, save_name, bins):
    angs = np.load(file)
    print(angs.shape)
    plt.figure()
    plt.hist(angs, bins=bins)
    plt.xlabel('Angles (normalized radians)')
    plt.ylabel('Counts')
    plt.savefig(save_name, dpi=200)

# plot the distribution of velocities
def plot_vel_distribution(file, save_name, bins):
    vels = np.load(file)
    print(vels.shape)
    plt.figure()
    plt.hist(vels, bins=bins)
    plt.xlabel('Velocities')
    plt.ylabel('Counts')
    plt.savefig(save_name, dpi=200)

def coor_to_angle_vel(y_file, vel_name, ang_name):
    y = np.load(y_file)
    vel = np.sqrt(np.sum(y**2, axis=1))
    # normalize the radians between -1 and 1
    ang = np.arctan2(y[:,1], y[:,0]) / math.pi
    print(vel.shape, ang.shape)
    np.save(vel_name, vel)
    np.save(ang_name, ang)

def plot_xy_hists(y_file, bins):
    y = np.load(y_file)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(y[:,0], bins=bins)
    plt.xlabel('x-coordinate')
    plt.ylabel('counts')

    plt.subplot(122)
    plt.hist(y[:,1], bins=bins)
    plt.xlabel('y-coordinate')
    plt.ylabel('counts')
    plt.savefig('xy_hists.png', dpi=200)

def plot_loss(history, epochs, model_name):
    plt.figure()
    plt.plot(np.arange(epochs), history.history['loss'], label='Train')
    plt.plot(np.arange(epochs), history.history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(IMG_PATH + '/' + model_name + '/loss_overall', dpi=200)
    # plt.show()

    plt.figure()
    plt.plot(np.arange(epochs), history.history['x_pred_loss'], label='Train x', color='blue')
    plt.plot(np.arange(epochs), history.history['val_x_pred_loss'], label='Validation x', color='blue', ls='--')
    plt.plot(np.arange(epochs), history.history['y_pred_loss'], label='Train y', color='red')
    plt.plot(np.arange(epochs), history.history['val_y_pred_loss'], label='Validation y', color='red', ls='--')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(IMG_PATH + '/' + model_name + '/loss_xy', dpi=200)
    # plt.show()

def plot_scatter(model, x_train, y_train, x_val, y_val, model_name):
    # predictions
    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)

    # visualise the velocity and directions of true vs prediction
    # i.e. on a 2d plane
    # train data
    plt.figure()
    plt.scatter(y_train[:,0], y_train[:,1], alpha=0.3, label='true')
    plt.scatter(y_pred_train[0], y_pred_train[1], alpha=0.3, label='pred')
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.title('Pred vs true on train data')
    plt.legend()
    plt.savefig(IMG_PATH + '/' + model_name + '/pred_v_true_train', dpi=200)

    # val data
    plt.figure()
    plt.scatter(y_val[:,0], y_val[:,1], alpha=0.3, label='true')
    plt.scatter(y_pred_val[0], y_pred_val[1], alpha=0.3, label='pred')
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.title('Pred vs true on val data')
    plt.legend()
    plt.savefig(IMG_PATH + '/' + model_name + '/pred_v_true_val', dpi=200)

    # visualise the x and y predictions separatelly
    # train data
    unity_x = np.linspace(-0.8,0.8,100)
    unity_y = np.linspace(-0.8,0.8,100)
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(unity_x, unity_x, c='orange', label='identity')
    plt.scatter(y_train[:,0], y_pred_train[0], alpha=0.3)
    plt.xlabel("True x"); plt.ylabel("Pred x")
    plt.legend()

    plt.subplot(122)
    plt.plot(unity_y, unity_y, c='orange', label='identity')
    plt.scatter(y_train[:,1], y_pred_train[1], alpha=0.3)
    plt.xlabel("True y"); plt.ylabel("Pred y")
    plt.legend()
    plt.savefig(IMG_PATH + '/' + model_name + '/pred_v_true_train_xy', dpi=200)

    # val data
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(unity_x, unity_x, c='orange', label='identity')
    plt.scatter(y_val[:,0], y_pred_val[0], alpha=0.3)
    plt.xlabel("True x"); plt.ylabel("Pred x")
    plt.legend()

    plt.subplot(122)
    plt.plot(unity_y, unity_y, c='orange', label='identity')
    plt.scatter(y_val[:,1], y_pred_val[1], alpha=0.3)
    plt.xlabel("True y"); plt.ylabel("Pred y")
    plt.legend()
    plt.savefig(IMG_PATH + '/' + model_name + '/pred_v_true_val_xy', dpi=200)

def plot_model(model_name):
    HOME_PATH = '/home/macleanlab/mufeng/models/'
    model_path = HOME_PATH + model_name
    cnn_model = keras.models.load_model(model_path)
    dot_img_file = 'model_diagram.png'
    tf.keras.utils.plot_model(cnn_model, to_file=dot_img_file, show_shapes=True, show_layer_names=True, dpi=200)
