## Front-end CNN
This is the front-end preprocesisng CNN for the trained SNN.

### How to generate spikes
The dot coherence change movies (in .tfrecord format) are in my folder in the server `macleanlab@205.208.22.225`, at `mufeng/NaturalMotionCNN/preprocessed`. If you are working in your own folder you can change the movie path in the `read_dot` function in `firing_rate.py` to the absolute path of this folder.

To generate spikes, you can clone this repository to your folder, then navigate to the `tf2gpu` virtual env on the 225 server:

```bash
conda activate tf2gpu
```

Then run the command 

```bash
CUDA_VISIBLE_DEVICES=1 python firing_rate.py 'ch_model4' 2
```

where `'ch_model4'` is the pre-trained CNN model, and `2` here means the number of dot coh change movies. `'ch_model4'` is the model trained with natural movies only, and there will soon be another model that is trained with a mixture of natural and artificial stimuli. The maximum number of dot coh change movies is 50, but later there will be more. There is a thrid optional argument for running this command, which is the number of natural movies you want to investigate. This is not needed for the purpose of generating spikes given artificial movies.

By running this command, a file `spike_train.npz` will be saved to your current working directory. This is a matrix of 0s and 1s, of size `num_frames x num_neurons`. It is saved as a scipy sparse matrix to save disk space, and to load and convert it to a numpy array or tensorflow tensor you can refer to [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz). You should be able to use this matrix for training the SNN.



