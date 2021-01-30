from tensorflow import keras
import tensorflow as tf
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

MODEL_PATH = cwd + 'models/'
IMG_PATH = cwd + 'plots/res_plots/'

parser = argparse.ArgumentParser(description='Args for the firing rates')
parser.add_argument('model_name', type=str, help='Name of the model, for better management')
parser.add_argument('num_dot_movies', type=int, help='Number of dot movies to check')
parser.add_argument('num_natural_movies', type=int, help='Number of natural movies to check')
args = parser.parse_args()
model_name = args.model_name
num_dot_movies = args.num_dot_movies
num_natural_movies = args.num_natural_movies

model = keras.models.load_model(MODEL_PATH+model_name)

def create_dataset(paths, max_seq_len=4800, encoding='png', pool=None):
    # again, you will change the features to reflect the variables in your own metadata
    # you may also change the max_seq_len (which is the maximum duration for each trial in ms)
    feature_description = {
        'frames': tf.io.VarLenFeature(tf.string),
        'change_label': tf.io.VarLenFeature(tf.int64),
        'coherence_label': tf.io.VarLenFeature(tf.int64),
        'direction_label': tf.io.VarLenFeature(tf.int64),
        'dominant_direction': tf.io.VarLenFeature(tf.int64),
        'trial_coherence': tf.io.VarLenFeature(tf.int64)
    }
    data_set = tf.data.TFRecordDataset(paths)

    def _parse_example(_x):
        _x = tf.io.parse_single_example(_x, feature_description)

        if encoding == 'png':
            _frames = tf.map_fn(lambda a: tf.io.decode_png(a), _x['frames'].values, dtype=tf.uint8)[:max_seq_len]

        _label = _x['coherence_label'].values
        _change_label = _x['change_label'].values
        _direction_label = _x['direction_label'].values
        _dominant_direction = _x['dominant_direction'].values
        _trial_coherence = _x['trial_coherence'].values

        return _frames, dict(tf_op_layer_coherence=_label, # 0 or 1, length 4800
                             tf_op_layer_change=_change_label, # 0 or 1, length 4800
                             tf_op_dir=_direction_label, # 1,2,3,4, length 4800
                             tf_dom_dir=_dominant_direction, # 1,2,3,4, length 10 (i.e. per trial/grey screen)
                             tf_trial_coh=_trial_coherence, # 0 or 1, length 10
                            )

    data_set = data_set.map(_parse_example, num_parallel_calls=24)
    return data_set

def plot_dot_response(all_movies):
    # get intermediate output
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.layers[12].output)
    # responses given one movie (containing 10 trials and 10 grey screens)                                      
    output = intermediate_layer_model(all_movies) # 20, 60, 16
    example = np.reshape(output, (-1, 16)) # 1200, 16
    # rec_res = tf.nn.relu(example).numpy()
    rec_res = example

    NUM_COLORS = rec_res.shape[1]
    cm = plt.get_cmap('nipy_spectral')
    cmap = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    fig = plt.figure(figsize=(12,12))
    for i in range(rec_res.shape[1]):
        ax = plt.subplot(rec_res.shape[1], 1, i+1)
        ax.set_ylim([-1,1])
        ax.plot(rec_res[:, i], alpha=1, c=cmap[i])
    fig.text(0.5, 0.06, 'time', ha='center')
    fig.text(0.06, 0.5, 'avg. firing rates over 4 frames', va='center', rotation='vertical')
    plt.savefig(IMG_PATH+'responses_dots.png', dpi=200)

    if False: # not-so-useful plots
        plt.figure(figsize=(12, 1))
        for i in range(rec_res.shape[1]):
            plt.plot(rec_res[:, i], alpha=0.6)
        plt.xlabel('time')
        plt.savefig(IMG_PATH+'responses_all.png', dpi=200)

def plot_natural_response(x):
    # get intermediate output
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.layers[12].output)                                 
    output = intermediate_layer_model(x) # 10, 60, 16
    output = np.reshape(output, (-1, 16)) # 600, 16
    # res = np.zeros((2*output.shape[0], output.shape[1]))
    # for i in range(0, 600, 60):
    #     res[2*i:2*i+60] = output[i:i+60]
    res = output

    NUM_COLORS = res.shape[1]
    cm = plt.get_cmap('nipy_spectral')
    cmap = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    fig = plt.figure(figsize=(12,12))
    for i in range(res.shape[1]):
        ax = plt.subplot(res.shape[1], 1, i+1)
        ax.set_ylim([-1,1])
        ax.plot(res[:, i], alpha=1, c=cmap[i])
    fig.text(0.5, 0.06, 'time', ha='center')
    fig.text(0.06, 0.5, 'avg. firing rates over 4 frames', va='center', rotation='vertical')
    plt.savefig(IMG_PATH+'responses_natural.png', dpi=200)

def prediction(all_movies):
    y_pred = model.predict(all_movies[::2])
    print(y_pred[0].shape)
    plt.figure()
    plt.scatter(y_pred[0], y_pred[1], alpha=0.3)
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.savefig(IMG_PATH+'/pred_dir', dpi=200)

def read_dot(num_movies):
    """
    Select a random subset of drifting dots movies

    num_movies: number of 4800 frames-long drifting dots movies to select. Each contains 10 trials and 10 grey screens
    """
    # each tfrecord file corresponds to only one movie
    file_names = [os.path.expanduser(f'preprocessed/processed_data_{i+1}.tfrecord') for i in range(num_movies)]
    data_set = create_dataset(file_names, 4800).batch(1)

    # ex[0] the frames (tensor), shape [1, 4800, 36, 64, 3]
    # ex[1] a dictionary, containing coh level, change label and direction
    k = 1
    movies = []
    true_dirs = []
    trial_cohs = []
    for ex in data_set: # iterate through num_movies
        trials = []
        print("Movie ", k)
        # the direction vector, of length max_seq_len, fixed for each trial/gray screen in the movie
        dirs = ex[1]['tf_op_dir'].numpy()[0]
        dom_dir = ex[1]['tf_dom_dir'].numpy()[0] # len = 10, direction vector of this movie
        trial_coh = ex[1]['tf_trial_coh'].numpy()[0] # len = 10, coh vector of this movie
        true_dirs.append(dom_dir)
        trial_cohs.append(trial_coh)
        # start and end frames of each trial movie and gray screen
        start_frames = np.arange(0, 80, 4)
        end_frames = np.arange(4, 84, 4)
        framerate = 60
        for i in range(2*len(dom_dir)): # 20
            trial = ex[0][:, start_frames[i]*framerate:end_frames[i]*framerate] # [1, 240, 36, 64, 3]
            trials.append(trial)

        # concatenate the trials and grey screens into a large tensor
        movie = tf.concat(trials, axis=0) # [20, 240, 36, 64, 3]
        movies.append(movie)
        k+=1
    
    all_movies = tf.concat(movies, axis=0) # [num_movies*20,240,36,64,3]
    directions = np.concatenate(true_dirs, axis=0) # shape=(num_movies*10,), [1,2,3,4] -> [0,90,180,270]
    coherences = np.concatenate(trial_cohs, axis=0) # shape=(num_movies*10,)

    return all_movies, directions, coherences

def read_natural(x_path, y_path, num_natural_movies):
    """
    num_natural_movies: number of natural movies (length=240 frames) to select
    """
    x = np.load(x_path, mmap_mode='r+')
    y = np.load(y_path, mmap_mode='r+')

    idx = np.random.randint(x.shape[0], size=num_natural_movies)
    return x[idx], y[idx]

def main(num_dot_movies, num_natural_movies):
    dot_movies, dot_directions, coherences = read_dot(num_dot_movies)
    natural_movies, natural_directions = read_natural('x_all.npy', 'y_all.npy', num_natural_movies)

    # plot the intermediate responses before the final projection
    plot_dot_response(dot_movies[0:20])
    plot_natural_response(natural_movies)

    print(dot_directions)
    print(coherences)
    angles = np.arctan2(natural_directions[:,1], natural_directions[:,0]) * 180 / np.pi
    distance = np.sqrt(natural_directions[:,0]**2 + natural_directions[:,1]**2)
    print(angles)
    print(distance)


if __name__ == '__main__':
    main(num_dot_movies, num_natural_movies)



