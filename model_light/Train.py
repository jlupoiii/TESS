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
import seaborn as sns

try:
    # bring in the fast cPickle and call it "pickle" so the code doesn't change
	import cPickle as pickle
except:
    # if cPickle isn't installed, just go ahead and use the slow pickle
	import pickle

print('Successfully imported all modules')

def train():

    # get data
    angle_folder = "//pdo//users//jlupoiii//TESS//model_light//angles//"
    ccd_folder = "//pdo//users//jlupoiii//TESS//model_light//ccds//"
    predictions_folder = "//pdo//users//jlupoiii//TESS//model_light//predictions//"

    # data matrices
    X = []
    Y = []
    ffis = []

    angles_dic = pickle.load(open(angle_folder+'angles_Oall_data.pkl', "rb"))

    for filename in os.listdir(ccd_folder):
#         print(filename)
        if len(filename) < 40 or filename[27] != '3': continue

        image_arr = pickle.load(open(ccd_folder+filename, "rb"))
        ffi_num = filename[18:18+8]
        try:
            angles = angles_dic[ffi_num]
        except:
            print('could not find ffi with number:', ffi_num)
            continue
        X.append(np.array([angles[10], angles[11], angles[18], angles[19], angles[22], angles[23], angles[24], angles[25]]))
        Y.append(image_arr.flatten())
        ffis.append(ffi_num)
#         print(ffi_num)

    X = np.array(X)
    Y = np.array(Y)
    ffis = np.array(ffis)
    print(X.shape, Y.shape)
    # separate data into testing and training
    x_train, x_test, y_train, y_test, ffis_train, ffis_test = train_test_split(X, Y, ffis, test_size = 0.2, random_state=4)
#     x_train, x_test, y_train, y_test = X, X, Y, Y    # for training on one datapoint only

    print('data loaded')
    
    # making and saving pairplot of input data
    pairplot_fig = sns.pairplot(pd.DataFrame(X, columns=['E3ez','E3az','M3ez','M3az','inv-ED','inv-MD','inv-squ-ED','inv-squ-MD']), height=1.5, corner=True).fig
    pairplot_fig.savefig("input_pairplot.png")
    plt.close()
    
    print('input data pairplot saved')
    
       
    # create model structure
    model = Sequential()
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(256))
    model.add(Dense(256))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    print("model initialized")

    # trains model
    history = model.fit(x_train, y_train, epochs = 4000, batch_size = 16, validation_data=(x_test,y_test))
    
    print("model finished")
    
    # save model
    model.save('model_dense')

    print("model saved")
    
    # plotting losses over epochs 
    plt.plot(range(10,len(history.history['loss'])), history.history['loss'][10:])
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.savefig('training_loss.png')
    plt.close()
    
    plt.plot(range(10,len(history.history['val_loss'])), history.history['val_loss'][10:])
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.savefig('validation_loss.png')
    plt.close()
    
    print("saved model history (loss graphs)")
    print('finished training')
    
  
    # predicting datapoints
    Y_hat_test = model.predict(x_test).reshape((y_test.shape[0], int(y_test.shape[1]**0.5), int(y_test.shape[1]**0.5)))
    Y_hat_train = model.predict(x_train).reshape((y_train.shape[0], int(y_train.shape[1]**0.5), int(y_train.shape[1]**0.5)))
    

    # plotting original vs model-predicted images for test set
    training_pts_to_view = 25
    test_pts_to_view = 25
    
    for _ in range(training_pts_to_view):
        i = random.randint(0, y_test.shape[0] - 1)
        y = y_test[i].reshape((int(y_test[0].shape[0]**0.5), int(y_test[0].shape[0]**0.5)))
        y_hat = Y_hat_test[i]
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(y, cmap='gray')
        axarr[1].imshow(y_hat, cmap='gray')
        axarr[0].title.set_text('Original')
        axarr[1].title.set_text('Prediction')
        f.suptitle('Test Datapoint \n ffi='+ffis_test[i]+' \n (E3ez, E3az, M3el, M3az, 1/ED, 1/MD, 1/ED^2, 1/MD^2)=\n'+str([np.format_float_positional
(s, precision=4, fractional=False) for s in x_test[i]]))
        plt.savefig(predictions_folder + ffis_test[i] + '_prediction_test.png')
        plt.close()
                

    # plotting original vs model-predicted images for training set
    for _ in range(test_pts_to_view):
        i = random.randint(0, y_train.shape[0] - 1)
        y = y_train[i].reshape((int(y_train[0].shape[0]**0.5), int(y_train[0].shape[0]**0.5)))
        y_hat = Y_hat_train[i]
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(y, cmap='gray')
        axarr[1].imshow(y_hat, cmap='gray')
        axarr[0].title.set_text('Original')
        axarr[1].title.set_text('Prediction')
        f.suptitle('Training Datapoint \n ffi='+ffis_train[i]+' \n (E3ez, E3az, M3el, M3az, 1/ED, 1/MD, 1/ED^2, 1/MD^2)=\n'+str([np.format_float_positional(s, precision=4, fractional=False) for s in x_train[i]]))
        plt.savefig(predictions_folder + ffis_train[i] + '_prediction_train.png')
        plt.close()
       
    print('saved all predicitons and its comparisons to the //predictions folder')    
        
    print('finished predicting')

if __name__=='__main__':
    train()
