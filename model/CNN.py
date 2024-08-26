import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

sys.path.append('/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/dataloader')
from dataloader import preprocess_data

class CNN_Model(object):
    def Build(self):
        w, h = 64, 64
        nodes = 64 
        self.model = models.Sequential()
        
        self.model.add(Input(shape=(w, h, 3), name='image_input'))

        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv2', padding='same'))
        self.model.add(MaxPooling2D((2, 2), name='pool2'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv3', padding='same'))
        self.model.add(MaxPooling2D((2, 2), name='pool3'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        return self.model

    def Compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
        return self.model
    
    def Train(self, input_set, output_set):
        X_train, X_test, Y_train, Y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)
        # early_callback = EarlyStopping(monitor="val_loss", min_delta= 0 , patience=10, verbose=1, mode="auto")
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_data=(X_test, Y_test), callbacks=[ tensorboard_callback])


model = CNN_Model()
model.Build()
model.Compile()
x, y = preprocess_data()
print(len(x), y.size)
model.Train(x, y)