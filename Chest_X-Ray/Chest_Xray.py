import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks

import os
import sys

#%%
def get_data(path_ini, img_size):
    """This function is for getting the chest xray imagines from the 
    specified path. The function also resizes the images.
    
    Args:
    path_ini (str): The directory of the images
    img_size (tuple): pixel x pixel image size
    
    Returns:
    np.array, np.array: Arrays of images and corresponding labels    
    """
    
    data = []
    labels = []
    for i in cases:
        path = path_ini + i
         
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img))
            resized_arr = cv2.resize(img_arr, (img_size[0], img_size[1]))
        
            data.append(resized_arr)
            labels.append(i[1:])                          
            
    return np.array(data), np.array(labels)
        


def scale_encode(X, y):
    
    """This function scales the images by 255 and encodes the image labels.
    
    Args:
    X (np.array): Array of images
    y (np.array): Array of labels
    
    Returns:
    np.array, np.array: Arrays of scaled images and corresponding 
                        encoded labels    
    """
    X_scl = np.reshape(X, X.shape)/255
    
    y_enc = np.array(pd.get_dummies(y, drop_first = True))
    
    return X_scl, y_enc

#%%
def image_augment(X, y, b_size):
    
    """This function is used for image augmentation.
    
    Args:
    X (np.array): Array of images
    y (np.array): Array of labels
    b_size (int): batch size
    
    Returns:
    Augmented image numpy array iterator  
    """

    datagen_train = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.2, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            shear_range=0.2,
            horizontal_flip = True,  # randomly flip images
            vertical_flip=False, # randomly flip images
            fill_mode='nearest')  
    
    # Note that the validation and test data should not be augmented!
    datagen_train.fit(X)
    train_aug = datagen_train.flow(X, y, batch_size = b_size)
    
    return train_aug


#%% Get images, scale, encode and augment

# define image paths and load the images
#the images are located in local pc due to large size.
#download the dataset from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
train_path = "D:\\Python_studies\\projects\\chest_xray_data\\archive\\chest_xray\\train"
test_path = "D:\\Python_studies\\projects\\chest_xray_data\\archive\\chest_xray\\test"
val_path = "D:\\Python_studies\\projects\\chest_xray_data\\archive\\chest_xray\\val"
cases = ["\\NORMAL", "\\PNEUMONIA"]

#resize the image to this size
img_size = (150, 150) 
Xtrain, ytrain = get_data(train_path, img_size)
Xtest, ytest = get_data(test_path, img_size)
Xval, yval = get_data(val_path, img_size)

# scale and encode the data
Xtrain_scl, ytrain_enc = scale_encode(Xtrain, ytrain)
Xval_scl, yval_enc = scale_encode(Xval, yval)
Xtest_scl, ytest_enc = scale_encode(Xtest, ytest)

#use image augmentation to get better representation for images
batch_size = 32 #batch size for image generator training
train_aug = image_augment(Xtrain_scl, ytrain_enc, batch_size)


#%%

def get_pretrained_model(base_model, in_size, n_last):
    
    """This function is used for getting initial pre-trained model for
    transfer learning.
    
    Args:
    base_model: base pre-trained model (see https://keras.io/api/applications/)
    in_size (tuple): size of the images (width, height, color channels)
    n_last (int): number of nuerons in the last layer before the output layer.
    
    Returns:
    Pre-trained model with head and final layers.
    """
    #Input shape = [width, height, color channels]
    inputs = layers.Input(shape = in_size)
    
    x = base_model(inputs)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_last, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    #Final Layer (Output)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=output)
    
    return model



def retrain_pretrained(model_pretrained, n_epochs):
    
    """This function is used for training the neural network model.
    
    Args:
    model_pretrained: Model with all layers added.
    n_epochs (int): number of epochs for the training.
    
    Returns:
    Trained model.
    
    Note: check parameters for the compile, EarlyStopping and 
         ReduceLROnPlateau
    """

    model_pretrained.compile(loss='binary_crossentropy',
                             optimizer = 
                             tf.keras.optimizers.Adam(learning_rate=5e-05), 
                             metrics='binary_accuracy')
    
    model_pretrained.summary()
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        min_delta=0.0000001,
        restore_best_weights=True)
    
    plateau = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor = 0.2,                                     
        patience = 2,                                   
        min_delt = 0.0000001,                                
        cooldown = 0,                               
        verbose = 1) 
    
    history = model_pretrained.fit(train_aug,
                                   epochs = n_epochs,
                                   validation_data= (Xval_scl,yval_enc),
                                   callbacks=[early_stopping, plateau]);

    #print(model.summary())

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return model_pretrained


#%% Model training

#input size (xpixels, ypixels, color channels)
in_size = Xtrain_scl.shape[1:]

#transfer learning will be used
#choose a base model
#VGG19 seems to work well
base_model = tf.keras.applications.VGG19(
    weights='imagenet',
    input_shape = in_size,
    include_top=False)

#we will unfreeze the whole layers and train the whole model again
base_model.trainable = True

#number of nuerons in the last layer before the output layer
n_last = 128 
#get pre-trained model
model_pretrained = get_pretrained_model(base_model, in_size, n_last)

#number of epochs to be used during training
n_epoch = 32
#train the model. Check some more parameters inside the function
#for learning rate, EarlyStopping and ReduceLROnPlateau
model_pretrained = retrain_pretrained(model_pretrained, n_epoch)

#%% use 0.5 as threshold 
train_pred = (model_pretrained.predict(Xtrain_scl)>0.5)
test_pred = (model_pretrained.predict(Xtest_scl)>0.5)

print("Accuracy score in train data %.2f" % accuracy_score(ytrain_enc, 
                                                           train_pred))

print("Accuracy score in test data %.2f" % accuracy_score(ytest_enc,
                                                          test_pred))

#%%
# Get the confusion matrix
test_pred2 = test_pred.ravel()
test_pred2 = np.where(test_pred2 == True, "PNEUMONIA","NORMAL")

cm  = confusion_matrix(ytest, test_pred2)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
