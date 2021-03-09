import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from keras.utils import np_utils
from keras.utils import to_categorical
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
import PIL
from PIL import Image
from keras.layers import Dense, BatchNormalization, Conv2D, Conv3D, Flatten
from keras import Model, Input
import os
import pandas as pd
from pandas import read_csv
from keras.models import model_from_json
import random
import imutils

#movies_directory contains the directory of movie files
#frames_directory is the directory that will contain more diretories
#Each of these directories will have individual frames from each movie
#Will additionally output a .csv file with three columns:
#Movie Name, First Frame, Number of Frames
#This csv defines movie clips.
#Example: ants4-2 2400 10 is the movie clip from ants4-2 starting at frame 2400 that is 10 frames long
def movies_to_frames(movies_directory, frames_directory, length, height, clip_length, out_csv=None, frames_flag=0):

    cwd = os.getcwd()

    #Setting up necessary structures for building the CSV
    movie_names = []
    first_frames = []
    clip_lengths = []

    name_head = 'stim' # to retain human order i.e. 1,2,3...
    for m in range(len(os.listdir(movies_directory))):
        movie = name_head + str(m+1) + '.mp4'
        aug_nums = 0
        randangles = random.sample(range(1, 359), aug_nums)
        # randangles = [0,0,0,0,0]
        #Process movie names
        movie_path = movies_directory + movie #This is where the movie comes from
        movie_name_orig, extension = os.path.splitext(movie)
        print(movie_name_orig)
        if extension != ".mp4":
            continue

        movie_dir_name = frames_directory + movie_name_orig #This is where we will output the frames

        #Making movie-specific directories, empty for now
        if not os.path.isdir(movie_dir_name):
            os.mkdir(movie_dir_name)
        os.chdir(movie_dir_name)

        cap = cv2.VideoCapture(movie_path)
        framerate = cap.get(5)

        #Some videos are not 60 FPS for some reason, so we will force 60
        #Some videos are not exactly 60 FPS, but near it.
        FPS_flag = 0
        frame_every = 0

        #FPS_flag = 0 if 60 FPS, 1 otherwise
        if framerate > 70:
            FPS_flag = 1
            frame_every = int(math.floor(framerate/60))

        frame_num = 0
        resize_dims = (length,height)


        #Goes through every frame of the movie at 60 FPS
        while(cap.isOpened()):
            #frame is the .jpg image. If you do anything like rotation, resizing, etc., perform the operation on 'frame'
            frame_ID = cap.get(1)
            ret, frame = cap.read()

            if(ret != True):
                break

            if FPS_flag == 0 or (FPS_flag == 1 and (frame_ID % frame_every == 0)):

                #Adds to the csv
                # change this condtion to include the first clips
                # if (frame_num != 0) and (frame_num % clip_length == 0):
                if frame_num % clip_length == 0:
                    movie_names.append(movie_name_orig + "_orig")
                    for i in range(aug_nums):
                        movie_names.append(movie_name_orig + "_rot{}".format(i+1))
                    first_frames.extend([frame_num]*(aug_nums+1))
                    clip_lengths.extend([clip_length]*(aug_nums+1))

                #Writes the original frame image to the proper directory, if the frames_flag is on
                file_name_orig = movie_name_orig + "_orig_frame{}.jpg".format(frame_num)
                file_names_rot = []
                for i in range(aug_nums):
                    file_name_rot = movie_name_orig + "_rot{}_frame{}.jpg".format(i+1, frame_num)
                    file_names_rot.append(file_name_rot)

                frame_num += 1

                #THIS IS WHERE FRAMES ARE WRITTEN
                if (frames_flag == 1):
                    # resized_frame = cv2.resize(frame, resize_dims)
                    # cv2.imwrite(file_name_orig, resized_frame)
                    # # horizontally flipped frames (flipping over y axis)
                    # horizflipped_frame = cv2.flip(resized_frame, 1)
                    # cv2.imwrite(file_name_fliphoriz, horizflipped_frame)
                    # # vertically flipped frames (flipping over x axis)
                    # vertflipped_frame = cv2.flip(resized_frame, 0)
                    # cv2.imwrite(file_name_flipvert, vertflipped_frame)
                    # # rotated frames
                    # rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
                    # cv2.imwrite(file_name_rotate, rotated_frame)

                    # rotate the (original frames) first, with random angles, then resize them
                    resized_frame = cv2.resize(frame, resize_dims)
                    # cv2.imwrite(file_name_orig, resized_frame)
                    cv2.imwrite(file_name_orig, frame)
                    for ang, file_name in zip(randangles, file_names_rot):
                        rotated_frame = imutils.rotate(resized_frame, angle=ang)
                        cv2.imwrite(file_name, rotated_frame)

        cap.release()

    dict = {'Movie Name': movie_names, 'First Frame': first_frames, 'Clip Length': clip_lengths}
    df = pd.DataFrame(dict)

    os.chdir(cwd)

    if out_csv == None:
        out_csv = cwd + "/Frame_Parameters_%d_Frame_Clips_dots.csv" % clip_length

    df = pd.DataFrame(dict)
    df.to_csv(out_csv, index=False)

    return out_csv


def compute_optical_flow(movie_name, first_index, num_frames, movies_directory):
    # now that we've added rotations, etc., the movie folder is not in fact the full name of the movie
    # find index of the underscore in movie_name
    underscore_idx = movie_name.find('_')
    # take all characters before that underscore to be movie_folder
    movie_folder = movie_name[0:underscore_idx]
    if(movies_directory[-1] == '/'):
        frames_directory = movies_directory + movie_folder + '/'
    else:
        frames_directory = movies_directory + '/' + movie_folder + '/'

    if not os.path.isdir(frames_directory):
        print("{} is not a valid movie or {} does not contain this movie".format(movie_name, movies_directory))
        raise

    try:
        first_frame_name = frames_directory + movie_name + "_frame{}.jpg".format(first_index)
        first_frame = cv2.imread(first_frame_name, cv2.IMREAD_GRAYSCALE)
        prev_frame = first_frame

    except:
        print("First frame number ({}) is more than number of frames".format(first_index))
        raise

    height = first_frame.shape[0]
    length = first_frame.shape[1]
    num_flows = num_frames-1
    flow_shape =(num_flows, height, length)
    final_index = first_index + num_frames - 1

    #In the computation of flow, pair of corresponding frames receives a flow field
    #Each flow field gives a vector to every single pixel
    #Hence, the shape of each of these arrays should be (num_frames-1, height,length)

    x_coords = np.zeros(flow_shape)
    y_coords = np.zeros(flow_shape)

    for frame_index in range(first_index, final_index):
        try:
            curr_frame_name = frames_directory + movie_name + "_frame{}.jpg".format(frame_index+1)
            curr_frame = cv2.imread(curr_frame_name, cv2.IMREAD_GRAYSCALE)

        except:
            print("Frame {} is out of bounds. Movie {} has {} frames".format(first_index, movie_name, len(os.listdir(frames_directory))))
            raise

        #using the Farneback method to estimate optical flow
        #IMPORTANT: This method uses consecutive frames to estimate flow
        #To get global motion, we will average vectors over space and time
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, flow =None, pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3,poly_n =5, poly_sigma=1.2, flags=0)
        x_coord = flow[..., 0]
        y_coord = flow[..., 1]

        x_coords[frame_index-first_index] = x_coord
        y_coords[frame_index-first_index] = y_coord

        prev_frame = curr_frame

    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    return mean_x, mean_y

#The input is a .csv file and an output path
#The input csv file should contain a 3 columns:
#Movie Name,First Frame, Clip Length
def optical_flows(csv_file, movies_directory, out_csv=None):
    movie_names = []
    first_indexes = []
    clip_lengths = []
    x_directions = []
    y_directions = []

    df = pd.read_csv(csv_file)
    num_movies = len(df['Movie Name'])
    for i in range(num_movies): # these include orig, fliphoriz, flipvert, and rotate
        movie_name = df['Movie Name'][i]
        first_index = df['First Frame'][i]
        clip_length = df['Clip Length'][i]

        x, y = compute_optical_flow(movie_name, first_index, clip_length, movies_directory)

        movie_names.append(movie_name)
        first_indexes.append(first_index)
        clip_lengths.append(clip_length)
        x_directions.append(x)
        y_directions.append(y)

    if out_csv == None:
        out_csv = os.getcwd() + "/flows_%d_frame_clips_dots.csv" %  clip_lengths[0]

    dict = {'Movie Name': movie_names, 'First Frame': first_indexes, 'Clip Length': clip_lengths,
            'x': x_directions, 'y': y_directions}
    df = pd.DataFrame(dict)
    df.to_csv(out_csv, index=False)

    return out_csv

def create_csvs():
    if (os.getcwd()[-1] == '/'):
        input_movies_directory = os.getcwd() + 'Movies_dot/'
        frames_directory = os.getcwd() + 'Frames_dot/'

    else:
        input_movies_directory = os.getcwd() + '/Movies_dot/'
        frames_directory = os.getcwd() + '/Frames_dot/'

    frame_parameters_csv = movies_to_frames(input_movies_directory, frames_directory, 64, 36, 240, frames_flag=1)
    # frame_parameters_csv = "/home/macleanlab/josh/NaturalMotionCNN/Frame_Parameters_240_Frame_Clips.csv"
    flows_csv = optical_flows(frame_parameters_csv, frames_directory)

create_csvs()

#------------------DATA PREPARATION---------------
#Converts movie clips to numpy arrays
#Elements of movies_directory should be directories
#Inside each directory will contain frames of a movie
#Output is tuple (movie_name, frames of the movie in order in a np array of dimensionality (num_frames, 36,64,3))
def movie_clip_to_arr(movies_directory, movie_name, first_index, height, length, clip_length):
    movie_folder = movie_name[0:movie_name.find('_')]
    frame_directory = movies_directory + movie_folder
    movie_data = np.zeros((clip_length,height,length,3))

    for frame_num in range(first_index, first_index+clip_length):
        frame_path = frame_directory + "/" +  movie_name + "_frame%d.jpg" % frame_num
        frame = Image.open(frame_path, mode='r')
        frame_data = np.asarray(frame, dtype='uint8')
        movie_data[frame_num-first_index] = frame_data
    movie_data = movie_data.astype('uint8')
    # movie_data = movie_data/255 #Normalization
    return (movie_name, movie_data)

#Creates the dataset, assuming that the frames have been processed and that the parameters file has been created
#x[i] will have shape (num_frames=240, 36, 64, 3)
#Therefore, x will have shape (num_movies, 240, 36, 64, 3)
#y has shape (num_movies, 2)
#y[i] = [x_direction, y_direction] of movie indexed with i
def regression_dataset(movies_directory, parameters_file, height, length, clip_length):
    df = pd.read_csv(parameters_file)
    num_movies = len(df['Movie Name'])
    x = np.zeros((num_movies, clip_length,height,length,3), dtype='uint8')
    y = np.zeros((num_movies, 2))

    for i in range(num_movies):
        movie_name = df['Movie Name'][i]
        print(movie_name)
        x_dir = df['x'][i]
        y_dir = df['y'][i]
        first_index = df['First Frame'][i]
        clip_len = df['Clip Length'][i]

        _, movie_data = movie_clip_to_arr(movies_directory, movie_name, first_index, height, length, clip_length)
        x[i] = movie_data
        y[i] = np.array([x_dir, y_dir])
    return x,y

# 1: 0; 2: 90; 3: 180; 4: 270
def categorical_dataset(movies_directory, parameters_file, height, length,clip_length):
    df = pd.read_csv(parameters_file)
    num_movies = len(df['Movie Name'])
    x = np.zeros((num_movies, clip_length,height,length,3), dtype='uint8')
    y = np.zeros(num_movies)

    for i in range(num_movies):
        movie_name = df['Movie Name'][i]
        x_dir = df['x'][i]
        y_dir = df['y'][i]
        first_index = df['First Frame'][i]
        clip_len = df['Clip Length'][i]

        _, movie_data = movie_clip_to_arr(movies_directory, movie_name, first_index, height, length, clip_length)
        x[i] = movie_data

        theta = np.arctan2(y_dir,x_dir)

        if theta >= -np.pi/8 and theta < np.pi/8:
            category = 0

        elif theta >= np.pi/8 and theta < 3*np.pi/8:
            category = 1

        elif theta >= 3*np.pi/8 and theta < 5*np.pi/8:
            category = 2

        elif theta >= 5*np.pi/8 and theta < 7*np.pi/8:
            category = 3

        elif theta >= 7*np.pi/16 or theta < -7*np.pi/8:
            category = 4

        elif theta >= -7*np.pi/8 and theta < -5*np.pi/8:
            category = 5

        elif theta >= -5*np.pi/8 and theta < -3*np.pi/8:
            category = 6

        elif theta >= -3*np.pi/8 and theta < -np.pi/8:
            category = 7

        y[i] = category

    return x,y

cwd = os.getcwd()

if(cwd[-1]=='/'):
    frames_directory = os.getcwd() + 'Frames_dot/'
    flows_csv = os.getcwd() + 'flows_240_frame_clips_dots.csv'
else:
    frames_directory = os.getcwd() + '/Frames_dot/'
    flows_csv = os.getcwd() + '/flows_240_frame_clips_dots.csv'

# #-----------------SETTING UP THE DATASET-----------------------
# #Use this if you want to try testing only for direction
# #x,y = categorical_dataset(frames_directory, flows_csv, 36, 64, 240)

x, y = regression_dataset(frames_directory, flows_csv, 36, 64, 240)
print(y.shape)
# x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.25, random_state=42)
# np.save('x_all_dot', x)
# np.save('y_all_dot', y)

np.savez_compressed('dot_data', x=x, y=y)