import numpy as numpy
import matplotlib.pyplot as plt
import os 
import cv2
from tqdm import tqdm

DATADIR = "/home/ali/Documents/Tensorflow Projects/Cats & Dogs/Cats-Dogs/Kaggle Cats and Dogs Dataset/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        # make array of pixels + RGB -> Grayscale conversion:
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
        img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE)) # Resize
        plt.imshow(img_array, cmap='gray')
        plt.show()
        print(img_array.shape)
        break
    break
