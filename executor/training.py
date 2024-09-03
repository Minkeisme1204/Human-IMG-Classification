import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

sys.path.append("/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/model")
sys.path.append('/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/')

from dataloader.dataloader import preprocess_data 
from model.CNN import CNN_Model

if __name__ == "__main__":
    model = CNN_Model()
    model.Build()
    model.Compile()
    x_train_test, y_train_test = preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test, test_size=0.2, random_state=42)
    print(len(x_train), y_train.size)
    model.Train(x_train, y_train)
    model.save('/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/results/Human_demo_1.h5')

    model.Evaluate(x_test, y_test)

    