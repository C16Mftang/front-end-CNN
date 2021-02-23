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
CUDA_VISIBLE_DEVICES=1 python firing_rate.py 'ch_model4' 50
```

where `'ch_model4'` is the pre-trained CNN model, and `50` here means the number of dot coh change movies. `'ch_model4'` is the model trained with natural movies only, and there will soon be another model that is trained with a mixture of natural and artificial stimuli. The maximum number of dot coh change movies is 60. 

here is a thrid optional argument for running this command, which is the number of natural movies you want to investigate. This is not needed for the purpose of generating spikes given artificial movies.

### Ouput
By running this command, a file `spike_train.npz` will be saved to your current working directory. This is a matrix of 0s and 1s, of size `[ms, num_neurons]`. The total number of `ms` is `num_movies x 20 x 4020`, where `20` is 10 trials plus 10 grey screens in each movie, and `4020` is the length in millisecond for each trial (and grey screen). In the case of the example command above, `num_movies` would be 50. The output matrix is saved as a scipy sparse matrix to save disk space, and to load and convert it to a numpy array or tensorflow tensor you can refer to [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz). 

Along with the matrix of spike trains, there will also be a file `changes.npy` saved to your working directory. This is a matrix of size `[num_movies, 10]`. Element `[i,j]` of this matrix stands for whether the coherence level changes in the j-th trial of the i-th movie. This would be the `y` (target variable) we use to train the SNN.




