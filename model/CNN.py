import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras import models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

sys.path.append('/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/')
from dataloader.dataloader import preprocess_data 

class CNN_Model(object):
    def Build(self):
        w, h = 64, 64
         
        self.model = models.Sequential()
        
        self.model.add(Input(shape=(w, h, 3), name='image_input'))

        self.model.add(Conv2D(8, (3, 3), activation='relu', name='conv1', padding='same'))
        # self.model.add(MaxPooling2D((3, 3), strides=(2,2), name='pool1'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(8, (3, 3), activation='relu', name='conv2', padding='same'))
        self.model.add(MaxPooling2D((3, 3), strides=(2,2), name='pool2'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(16, (3, 3), activation='relu', name='conv3', padding='same'))
        self.model.add(MaxPooling2D((3, 3), strides=(2,2), name='pool3'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3, 3), activation='relu', name='conv4', padding='same'))
        # self.model.add(MaxPooling2D((3, 3), strides=(2,2), name='pool4'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3, 3), activation='relu', name='conv5', padding='same'))
        self.model.add(MaxPooling2D((3, 3), strides=(2,2), name='pool5'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))

        # self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dropout(0.3))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        return self.model

    def Compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
        return self.model
    
    def Train(self, input_set, output_set):
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)
        # early_callback = EarlyStopping(monitor="val_loss", min_delta= 0 , patience=10, verbose=1, mode="auto")
        log_dir = "/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val), callbacks=[tensorboard_callback])
        

    def Evaluate(self, X_test, Y_test):
        metrics = self.model.evaluate(X_test, Y_test, batch_size = 64)
        print("Loss value is: {}".format(metrics[0]))
        print("Accuracy value is: {}".format(metrics[1]))
        print("Recall value is: {}".format(metrics[2]))
        print("Precision value is: {}".format(metrics[3]))
        print("AUC value is: {}".format(metrics[4]))
        return metrics
    def save(self, filename):
        self.model.save(filename)
        pass



