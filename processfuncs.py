import os
import sys
import keras
import numpy as np
import tensorflow as tf


def preprocess_image(img_path,nrows,ncols):
    img = keras.preprocessing.image.load_img(img_path,target_size=(nrows,ncols))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(y,nrows,ncols):
    y = y.reshape((nrows,ncols,3))
    y[:,:,0] += 103.939
    y[:,:,1] += 116.779
    y[:,:,2] += 123.68
    y = y[:,:,::-1]
    y = np.clip(y,0,255).astype("uint8")
    return y