from tensorflow import keras
import tensorflow as tf
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_pred_dots
import scipy.sparse

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

MODEL_PATH = cwd + 'models/'
IMG_PATH = cwd + 'plots/res_plots/'

parser = argparse.ArgumentParser(description='Args for generating the spikes')
parser.add_argument('model_name', type=str, help='Name of the model')
parser.add_argument('num_dot_movies', type=int, help='Number of dot movies to check')
parser.add_argument('--num_natural_movies', default=20, type=int, help='Number of natural movies to check')
args = parser.parse_args()
model_name = args.model_name
num_dot_movies = args.num_dot_movies
num_natural_movies = args.num_natural_movies

FRAMES_PER_TRIAL = 240
FRAMERATE = 60
MS_PER_TRIAL = (FRAMES_PER_TRIAL // FRAMERATE) * 1000

model = keras.models.load_model(MODEL_PATH+model_name)

def create_dataset(paths, max_seq_len=4800, encoding='png', pool=None):
    """
    Read in .tfrecord datasets of the drifting dots movies
    """
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

def read_dot(num_movies):
    """
    Select a random subset of drifting dots movies
    num_movies: number of 4800 frames-long drifting dots movies to select. Each contains 10 trials and 10 grey screens
    """
    # each tfrecord file corresponds to only one movie (4800 frames)
    file_names = [os.path.expanduser(f'preprocessed/processed_data_{i+1}.tfrecord') for i in range(num_movies)]
    data_set = create_dataset(file_names, 4800).batch(1)

    # ex[0] the frames (tensor), shape [1, 4800, 36, 64, 3]
    # ex[1] a dictionary, containing coh level, change label and direction
    k = 1
    movies = []
    true_dirs = []
    trial_cohs = []
    for ex in data_set:
        trials = []
        print("Movie ", k)
        # the direction vector, of length max_seq_len, fixed for each trial/gray screen in the movie
        dirs = ex[1]['tf_op_dir'].numpy()[0]
        dom_dir = ex[1]['tf_dom_dir'].numpy()[0] # len = 10, direction vector of this movie
        trial_coh = ex[1]['tf_trial_coh'].numpy()[0] # len = 10, coh vector of this movie
        true_dirs.append(dom_dir)
        trial_cohs.append(trial_coh)
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
    Read in natural motion movies 
    num_natural_movies: number of natural movies (length=240 frames) to select
    """
    x = np.load(x_path, mmap_mode='r+')
    y = np.load(y_path, mmap_mode='r+')

    idx = np.random.randint(x.shape[0], size=num_natural_movies)
    return x[idx], y[idx]

def plot_firing_rates(all_movies, stim='natural'):
    """
    Plot the 16 neurons' firing rates to drifting dots or natural movies, rectified
    all_movies: input
    stim: 'natural' or 'dots'
    """
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.layers[12].output)
    # responses given one movie                    
    output = intermediate_layer_model(all_movies) # 20, 60, 16
    example = np.reshape(output, (-1, 16)) # 1200, 16
    rec_res = tf.nn.relu(example).numpy()

    NUM_COLORS = rec_res.shape[1]
    cm = plt.get_cmap('nipy_spectral')
    cmap = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    fig = plt.figure(figsize=(12,12))
    for i in range(rec_res.shape[1]):
        ax = plt.subplot(rec_res.shape[1], 1, i+1)
        ax.set_ylim([0, 1])
        ax.plot(rec_res[:, i], alpha=1, c=cmap[i])
    fig.text(0.5, 0.06, 'time', ha='center')
    fig.text(0.06, 0.5, 'avg. firing rates over 4 frames', va='center', rotation='vertical')
    plt.savefig(IMG_PATH+'responses_'+stim, dpi=200)

    if False: # plot all neurons in one frame
        plt.figure(figsize=(12, 1))
        for i in range(rec_res.shape[1]):
            plt.plot(rec_res[:, i], alpha=0.6)
        plt.xlabel('time')
        plt.savefig(IMG_PATH+'responses_all.png', dpi=200)

def plot_dot_predictions():
    """
    Plot true optic flow vs predicted optic flow by the model
    """
    plot_pred_dots(model)

def spike_generation(all_movies):
    """
    Generate spikes based on firing rates
    Output: (num_dot_movies*20, 240, 16), binary 
    """
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.layers[12].output)
    # responses given one movie                    
    output = intermediate_layer_model(all_movies) # 20*num_dot_movies, 60, 16
    f_rates = tf.nn.relu(output).numpy()
    num_neurons = f_rates.shape[2]

    f_rates_r = np.repeat(f_rates, int(MS_PER_TRIAL/f_rates.shape[1])+1, axis=1) # 20*num_dot_movies, 4020, 16
    
    # random matrix between [0,1] for spike generation
    random_matrix = np.random.rand(f_rates_r.shape[0], f_rates_r.shape[1], num_neurons)
    spikes = (f_rates_r - random_matrix > 0)*1
    return spikes

def raster_plot(all_movies):
    """
    Generate raster plot based on spike trains
    """
    spikes = spike_generation(all_movies)
    example = np.reshape(spikes, (-1, 16)) # 4800, 16

    plt.figure()
    for i in range(example.shape[0]):
        for j in range(example.shape[1]):
            if example[i,j] == 1:
                x1 = [i,i]
                x2 = [j-0.25,j+0.25]
                plt.plot(x1, x2, color='black', linewidth=0.2)
    plt.xlabel('Time (frames)')
    plt.ylabel('Neurons')
    plt.savefig(IMG_PATH+'raster_plot.png', dpi=200)
                
def main(num_dot_movies, num_natural_movies):
    """
    Difference between two inputs:

    num_dot_movies: number of drifting dot movies, each containing 10 moving dots and 10 grey screens, total length = 4800 frames
    num_natural_movies: number of natural movies, total length = 240 frames

    The binary spike train matrix is saved as a scipy sparse matrix (.npz)
    """
    dot_movies, dot_directions, coherences = read_dot(num_dot_movies)
    spikes = spike_generation(dot_movies) # 20*num_dot_movies*4020, 16
    print(spikes.shape)
    spikes_sparse = scipy.sparse.csc_matrix(spikes.reshape((-1, spikes.shape[2])))

    scipy.sparse.save_npz('spike_train.npz', spikes_sparse)
    
    if False: # generate plots
        plot_firing_rates(dot_movies[0:20], stim='dots') # plot using the first drifting dots movie
        raster_plot(dot_movies[0:20])

        natural_movies, natural_directions = read_natural('x_all.npy', 'y_all.npy', num_natural_movies)
        plot_firing_rates(natural_movies, stim='natural')
        plot_dot_predictions()
        angles = np.arctan2(natural_directions[:,1], natural_directions[:,0]) * 180 / np.pi
        distance = np.sqrt(natural_directions[:,0]**2 + natural_directions[:,1]**2)

if __name__ == '__main__':
    main(num_dot_movies, num_natural_movies)



