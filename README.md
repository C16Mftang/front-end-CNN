## Front-end CNN
This is the front-end preprocesisng CNN for the trained SNN.

## How to generate spikes
### Input
The input to the pre-trained CNN model is the raw dot coherence change movies. These movies (in .tfrecord format) are in my folder in the server `macleanlab@205.208.22.225`, at `mufeng/tfrecord_data_processing/preprocessed`. The directory in `firing_rate.py` to these movies has been changed to the absolute path, so you should be able to run the code directly.

To generate spikes, you can clone this repository to your folder, then navigate to the `tf2gpu` virtual env on the `.225` server:

```bash
conda activate tf2gpu
```

Then run the command 

```bash
CUDA_VISIBLE_DEVICES=1 python firing_rate.py 'ch_model4' 60
```

where `'ch_model4'` is the pre-trained CNN model, and `50` here means the number of dot coh change movies. `'ch_model4'` is the model trained with natural movies only, and there will soon be another model that is trained with a mixture of natural and artificial stimuli. The maximum number of dot coh change movies is 60. 

There is a thrid optional argument for running this command, which is the number of natural movies you want to investigate. This is not needed for the purpose of generating spikes given artificial movies.

### Ouput
By running the script, a folder `CNN_outputs` will be created in your working directory, which contains 3 files:

1. `spike_train.npz`: this is a matrix of spike trains generated from the firing rates. Its size is `[num_moviesx10x4080, num_neurons]`. In the example above, `num_movies` would be 50, and each movie contains 10 trials (we have excluded grey screens) - that's what the `10` means. `4080` here means the length of a trial (or a grey screen) in milliseconds, and it results from repeating the firing rates at each of the compressed 60 timesteps 68 times (60*68=4080). Therefore the `num_moviesx10x4080` is the total length of all the movies in milliseconds. The `num_neurons` here is fixed at 16. When training the SNN, we feed the spike trains into the SNN trial by trial, so you should reshape this matrix into `[num_moviesx10, 4080, num_neurons]` so each of your input instance would be a 4080(ms)x16(cells) matrix. This matrix is also saved as a scipy sparse matrix to save disk space, and to load and convert it to a numpy array or tensorflow tensor you can refer to [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz). 

2. `coherences.npz`: this is a matrix of coherence levels (0 for 15% and 1 for 100%), of size `[num_movies, 40800]`. Again, in the example above `num_movies` would be 50, and `40800` is the total length (in milliseconds) of a movie containing 10 trials. This matrix records the coherence level at each millisecond, in each movie. If you are training your SNN to predict the ms-by-ms coherence levels, you should load this as your target variable. (It is a scipy sparse matrix as well!)

3. `changes.npy`: this is a matrix that records whether there is a coherence change in each trial, of size `[num_movies, 10]` (i.e. each movie contains 10 trials). If you want to train your SNN to detect change(1)/no change(0) in the trials, you should load this as your target variable.





