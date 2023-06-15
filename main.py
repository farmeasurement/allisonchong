import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

def convertImage(filePath): 
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr

#Convert train data to tensor
trainDataDf = pd.read_csv("book30-listing-train.csv", header=None, usecols=[1,6], encoding_errors='replace')
trainDataDf[1] = '224x224/' + trainDataDf[1].astype(str) 
n = 5130
list_df = [trainDataDf[i:i+n] for i in range(0,trainDataDf.shape[0],n)]
for i in range(11): 
    list_df[i].iloc[:,0] = list_df[i].iloc[:,0].apply(convertImage)

"""train_images = tf.convert_to_tensor(trainDataDf[1])
train_labels = tf.convert_to_tensor(trainDataDf[6])"""
"""
testDataDf = pd.read_csv("book30-listing-test.csv", header=None, usecols=[1,6], encoding_errors='replace')
testDataDf[1] = '224x224/' + testDataDf[1].astype(str)"""
"""test_images = tf.convert_to_tensor(testDataDf[1])
test_labels = tf.convert_to_tensor(testDataDf[6])"""

