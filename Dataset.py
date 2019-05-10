import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
from tqdm import tqdm
import random # for shuffling training data
import pickle # for serializing the data before saving to disk

DATADIR = "/home/ali/Documents/Tensorflow Projects/Cats & Dogs/Cats-Dogs/Kaggle Cats and Dogs Dataset/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100

# Making Training Data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category) # 0=dog, 1=cat

        for img in os.listdir(path):
            try:
                # make array of pixels + RGB -> Grayscale conversion:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE)) # Resize
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
                #print("Exception thrown: ",e)
    random.shuffle(training_data)

create_training_data()
# Checking the length and randomness
# for s in training_data[:10]:
#     print(s[1])
# print(len(training_data))

### Making model
X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

# print(X[0].reshape(-1,IMG_SIZE,IMG_SIZE,1))

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

### Save data with pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
# To open:
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

    
            
        
