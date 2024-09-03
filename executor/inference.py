import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import numpy as np
path = '/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/human data for classification/Inference'
model = load_model('/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/results/Human_demo_1.h5')

img_list = os.listdir(path)
image_size = (64 ,64)

for img_path in img_list:
    full_img_path = os.path.join(path, img_path)
    img = image.load_img(full_img_path, target_size = image_size, color_mode = 'rgb')
    img = image.img_to_array(img)
    temp = img/255
    img = np.expand_dims(img, axis=0)  # Add one dimension for batch size
    img = img/255.
    result = model.predict(img)
    # result = np.argmax(result, axis=1)[0]
    print(result)

    if result > 0.5:
        result = 'Human'
    else:
        result = 'Non-Human'
    print(result)
    plt.imshow(temp)
    plt.title("inference classification")
    plt.show()