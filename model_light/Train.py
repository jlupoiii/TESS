################################
###  PYTORCH IMPLEMENTATION  ###
################################

# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import seaborn as sns
from sklearn.model_selection import train_test_split

try:
    # bring in the fast cPickle and call it "pickle" so the code doesn't change
	import cPickle as pickle
except:
    # if cPickle isn't installed, just go ahead and use the slow pickle
	import pickle

print('Successfully imported all modules')


# Neural Network Model Class
class Dense_NN(nn.Module):
    def __init__(self):
        super(Dense_NN, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 256)
        self.fc4 = nn.Linear(256, 256)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, num_epochs):
        training_loss = []
        validation_loss = []
        
        for epoch in range(num_epochs):
            
            X, Y = next(iter(train_dataloader))
            
            # Forward pass
            Y_hat = model(X)
            loss_t = loss_fn(Y_hat, Y)

            # Backward pass
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            
            # training loss
            training_loss.append(float(loss_t))
            
            # validation loss
            X_test, Y_test = next(iter(test_dataloader))
            Y_test_hat = model(X_test)
            loss_v = loss_fn(Y_test_hat, Y_test)
            validation_loss.append(float(loss_v))
            
            print('Epoch ' + str(epoch+1) + '/' + str(num_epochs) + '  Training loss: ' + str(float(loss_t))[:7] + '  Validation loss: ' + str(float(loss_v))[:7])
        
        return training_loss, validation_loss




def train_NN():
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
        if len(filename) < 40 or filename[27] != '3': continue

        image_arr = pickle.load(open(ccd_folder+filename, "rb"))
        ffi_num = filename[18:18+8]
        try:
            angles = angles_dic[ffi_num]
            print('Got ffi number', ffi_num)
        except:
            print('Could not find ffi with number:', ffi_num)
            continue
        X.append(np.array([angles[10], angles[11], angles[18], angles[19], angles[22], angles[23], angles[24], angles[25]]))
        Y.append(image_arr.flatten())
        ffis.append(ffi_num)

    X = np.array(X)
    Y = np.array(Y)
    ffis = np.array(ffis)
    
    # making and saving pairplot of input data
    pairplot_fig = sns.pairplot(pd.DataFrame(X, columns=['E3ez','E3az','M3ez','M3az','inv-ED','inv-MD','inv-squ-ED','inv-squ-MD']), height=1.5, corner=True).fig
    pairplot_fig.savefig("input_pairplot.png")
    plt.close()
    
    print('input data pairplot saved')
    
    # separate data into testing and training, than make into tensors, datasets, and daatloaders
    x_train, x_test, y_train, y_test, ffis_train, ffis_test = train_test_split(X, Y, ffis, test_size = 0.2, random_state=4)
    x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = torch.Tensor(x_train), torch.Tensor(x_test), torch.Tensor(y_train), torch.Tensor(y_test)
    
    dataset_train = TensorDataset(x_train_tensor,y_train_tensor)
    dataset_test = TensorDataset(x_test_tensor,y_test_tensor)
    
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True) 

    print('data loaded')
    
    ##############################################
    
    # create model structure
    model = Dense_NN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    print("model initialized")

    # Train the model and save training/validation loss
    training_loss, validation_loss = model.train(dataloader_train, dataloader_test, loss_fn, optimizer, num_epochs=10000)
    
    print("model trained")
    
    # plotting losses over epochs 
    plt.plot(range(10,len(training_loss)), training_loss[10:])
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.savefig('training_loss.png')
    plt.close()
    
    plt.plot(range(10,len(validation_loss)), validation_loss[10:])
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.savefig('validation_loss.png')
    plt.close()
    
    print("saved model history (loss graphs)")
    
    # save model's state dictionary
    torch.save(model.state_dict(), 'model_dense_pytorch.pth')

    print("model saved")
    
    
    
    # predicting datapoints
    Y_hat_test = model(x_test_tensor).detach().numpy().reshape((y_test.shape[0], int(y_test.shape[1]**0.5), int(y_test.shape[1]**0.5)))
    Y_hat_train = model(x_train_tensor).detach().numpy().reshape((y_train.shape[0], int(y_train.shape[1]**0.5), int(y_train.shape[1]**0.5)))
    
    print(Y_hat_test.shape)
    print(Y_hat_train.shape)
    
    ##############################################

    # plotting original vs model-predicted images for test set
    training_pts_to_view = 50
    test_pts_to_view = 50
    
    for _ in range(training_pts_to_view):
        print(_)
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
        print(_)
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
    train_NN()




    
    
    
    
#########################################
###  TENSORFLOW KERAS IMPLEMENTATION  ###
#########################################


# # import dependencies
# import tensorflow
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Convolution2D, MaxPooling2D
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# from PIL import Image
# import random
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# try:
#     # bring in the fast cPickle and call it "pickle" so the code doesn't change
# 	import cPickle as pickle
# except:
#     # if cPickle isn't installed, just go ahead and use the slow pickle
# 	import pickle

# print('Successfully imported all modules')

# def train_NN():

#     # get data
#     angle_folder = "//pdo//users//jlupoiii//TESS//model_light//angles//"
#     ccd_folder = "//pdo//users//jlupoiii//TESS//model_light//ccds//"
#     predictions_folder = "//pdo//users//jlupoiii//TESS//model_light//predictions//"

#     # data matrices
#     X = []
#     Y = []
#     ffis = []

#     angles_dic = pickle.load(open(angle_folder+'angles_Oall_data.pkl', "rb"))

#     for filename in os.listdir(ccd_folder):
# #         print(filename)
#         if len(filename) < 40 or filename[27] != '3': continue

#         image_arr = pickle.load(open(ccd_folder+filename, "rb"))
#         ffi_num = filename[18:18+8]
#         try:
#             angles = angles_dic[ffi_num]
#         except:
#             print('could not find ffi with number:', ffi_num)
#             continue
#         X.append(np.array([angles[10], angles[11], angles[18], angles[19], angles[22], angles[23], angles[24], angles[25]]))
#         Y.append(image_arr.flatten())
#         ffis.append(ffi_num)
# #         print(ffi_num)

#     X = np.array(X)
#     Y = np.array(Y)
#     ffis = np.array(ffis)
#     print(X.shape, Y.shape)
#     # separate data into testing and training
#     x_train, x_test, y_train, y_test, ffis_train, ffis_test = train_test_split(X, Y, ffis, test_size = 0.2, random_state=4)
# #     x_train, x_test, y_train, y_test = X, X, Y, Y    # for training on one datapoint only

#     print('data loaded')
    
#     # making and saving pairplot of input data
#     pairplot_fig = sns.pairplot(pd.DataFrame(X, columns=['E3ez','E3az','M3ez','M3az','inv-ED','inv-MD','inv-squ-ED','inv-squ-MD']), height=1.5, corner=True).fig
#     pairplot_fig.savefig("input_pairplot.png")
#     plt.close()
    
#     print('input data pairplot saved')
    
#     ##############################################
    
#     # create model structure
#     model = Sequential()
#     model.add(Dense(100))
#     model.add(Dense(100))
#     model.add(Dense(256))
#     model.add(Dense(256))
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
#     print("model initialized")

#     # trains model
#     history = model.fit(x_train, y_train, epochs = 4000, batch_size = 16, validation_data=(x_test,y_test))
    
#     print("model finished")
    
#     # save model
#     model.save('model_dense')

#     print("model saved")
    
#     # plotting losses over epochs 
#     plt.plot(range(10,len(history.history['loss'])), history.history['loss'][10:])
#     plt.title('Training Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (MSE)')
#     plt.yscale('log')
#     plt.savefig('training_loss.png')
#     plt.close()
    
#     plt.plot(range(10,len(history.history['val_loss'])), history.history['val_loss'][10:])
#     plt.title('Validation Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (MSE)')
#     plt.yscale('log')
#     plt.savefig('validation_loss.png')
#     plt.close()
    
#     print("saved model history (loss graphs)")
#     print('finished training')
    
    
#     # predicting datapoints
#     Y_hat_test = model.predict(x_test).reshape((y_test.shape[0], int(y_test.shape[1]**0.5), int(y_test.shape[1]**0.5)))
#     Y_hat_train = model.predict(x_train).reshape((y_train.shape[0], int(y_train.shape[1]**0.5), int(y_train.shape[1]**0.5)))
    
#     ##############################################

#     # plotting original vs model-predicted images for test set
#     training_pts_to_view = 100
#     test_pts_to_view = 100
    
#     for _ in range(training_pts_to_view):
#         i = random.randint(0, y_test.shape[0] - 1)
#         y = y_test[i].reshape((int(y_test[0].shape[0]**0.5), int(y_test[0].shape[0]**0.5)))
#         y_hat = Y_hat_test[i]
        
#         f, axarr = plt.subplots(1,2)
#         axarr[0].imshow(y, cmap='gray')
#         axarr[1].imshow(y_hat, cmap='gray')
#         axarr[0].title.set_text('Original')
#         axarr[1].title.set_text('Prediction')
#         f.suptitle('Test Datapoint \n ffi='+ffis_test[i]+' \n (E3ez, E3az, M3el, M3az, 1/ED, 1/MD, 1/ED^2, 1/MD^2)=\n'+str([np.format_float_positional
# (s, precision=4, fractional=False) for s in x_test[i]]))
#         plt.savefig(predictions_folder + ffis_test[i] + '_prediction_test.png')
#         plt.close()
                

#     # plotting original vs model-predicted images for training set
#     for _ in range(test_pts_to_view):
#         i = random.randint(0, y_train.shape[0] - 1)
#         y = y_train[i].reshape((int(y_train[0].shape[0]**0.5), int(y_train[0].shape[0]**0.5)))
#         y_hat = Y_hat_train[i]
        
#         f, axarr = plt.subplots(1,2)
#         axarr[0].imshow(y, cmap='gray')
#         axarr[1].imshow(y_hat, cmap='gray')
#         axarr[0].title.set_text('Original')
#         axarr[1].title.set_text('Prediction')
#         f.suptitle('Training Datapoint \n ffi='+ffis_train[i]+' \n (E3ez, E3az, M3el, M3az, 1/ED, 1/MD, 1/ED^2, 1/MD^2)=\n'+str([np.format_float_positional(s, precision=4, fractional=False) for s in x_train[i]]))
#         plt.savefig(predictions_folder + ffis_train[i] + '_prediction_train.png')
#         plt.close()
       
#     print('saved all predicitons and its comparisons to the //predictions folder')    
        
#     print('finished predicting')

# if __name__=='__main__':
#     train_NN()