# import dependencies
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from tensorflow import keras
import pandas as pd
import numpy as np
from PIL import Image
import random
import os
import matplotlib.pyplot as plt

try:
    # bring in the fast cPickle and call it "pickle" so the code doesn't change
	import cPickle as pickle
except:
    # if cPickle isn't installed, just go ahead and use the slow pickle
	import pickle

print('Successfully imported all modules')



def predict():
    # loads model
    model = load_model('model_dense')
    
    # get data
    angle_folder = "//pdo//users//jlupoiii//model_light//angles//"
    ccd_folder = "//pdo//users//jlupoiii//model_light//ccds//"
    predictions_folder = "//pdo//users//jlupoiii//model_light//predictions//"

    angles_dic = pickle.load(open(angle_folder+'angles_O13_data.pkl', "rb"))
    
    X = []
    Y = []
    ffis = []
    
    for filename in os.listdir(ccd_folder):
        if filename[27] != '3': continue

        image_arr = pickle.load(open(ccd_folder+filename, "rb"))
        ffi_num = filename[18:18+8]
        ffis.append(ffi_num)
        angles = angles_dic[ffi_num]
        X.append(np.array([angles[10], angles[11], angles[18], angles[19], angles[22], angles[23]]))
#         print(ffi_num)
        Y.append(image_arr)

    X = np.array(X)
    Y = np.array(Y)
    
#     print(Y.shape, type(Y.shape))
    
    Y_hat = model.predict(X).reshape((Y.shape[0], Y.shape[1], Y.shape[1]))
    
    for i in range(len(ffis)):
        y = Y[i]
        y_hat = Y_hat[i]
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(y, cmap='gray')
        axarr[1].imshow(y_hat, cmap='gray')
        axarr[0].title.set_text('Original')
        axarr[1].title.set_text('Prediction')
        plt.savefig(predictions_folder + ffis[i] + '_prediction.png')
        plt.close()
        print(i)
        
        if i==100: break
        
        
    print('saved all predicitons and its comparisons to the //predictions folder')    
        
    print('done')



if __name__=="__main__":
    predict()
    