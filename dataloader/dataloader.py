import os
import numpy as np 
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from sklearn.model_selection import train_test_split

image_size = (64, 64)
def label_image_data(data_path, label = None):
    list_image_data = os.listdir(data_path)
    image_data = []
    labelled_list = []

    for img_path in list_image_data:
        full_img_path = os.path.join(data_path, img_path)
        img = image.load_img(full_img_path, target_size = image_size, color_mode = 'rgb')
        img = image.img_to_array(img)
        img = img/255
        labelled_list.append(label)
        image_data.append(img)
    
    img = np.array(img)
    print(len(image_data), len(labelled_list))
    return image_data, labelled_list

def preprocess_data():
    human_path = '/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/human data for classification/Human'
    non_human_path = '/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/human data for classification/Non-Human'

    human_img, human_lanels = label_image_data(human_path, label=1)
    non_human_img, non_human_labels = label_image_data(non_human_path, label=0)

    img_data = np.array(human_img + non_human_img)

    labels_data = np.array(human_lanels + non_human_labels)

    # img_data = np.array(human_img + non_human_img)
    # labels_data = np.array(human_lanels + non_human_labels)

    return img_data, labels_data

