## Front-end CNN
This is the front-end preprocesisng CNN for the trained SNN.

### How to generate spikes
Before everything, you should get a copy of the dot coherence change movies (in .tfrecord format) in my folder in `macleanlab@205.208.22.225`, at `mufeng/NaturalMotionCNN/preprocessed`. Copying the `preprocessed` folder should work.

After setting up the stimulus movies, you can clone this repository to your folder, then navigate to the `tf2gpu` virtual env:

```bash
conda activate tf2gpu
```

Then run the command 

```bash
CUDA_VISIBLE_DEVICES=1 python firing_rate.py 'ch_model4' 2
```

where `'ch_model4'` is the pre-trained CNN model, and `2` here means the number of dot coh change movies. Currently, only the model `'ch_model4'` works, and it is already included in the repo. The maximum number of dot coh change movies is 50, but later there will be more. There is a thrid optional argument for running this command, which is the number of natural movies you want to investigate; for our purpose, it is not needed.

By running this command, a file `spike_train.npz` will be saved to your current working directory. This is a matrix of 0s and 1s, of size `num_frames x num_neurons`. It is saved as a scipy sparse matrix to save disk space, and to load and convert it to a numpy array or tensorflow tensor you can refer to [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz). You should be able to use this matrix for training the SNN.



