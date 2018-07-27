import numpy as np
import pandas as pd

dataset = pd.read_csv('meta-data/test.csv')

y = dataset.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features= [0])
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()

with open("test.txt","r") as f:
    input_images= f.readlines()

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

model = load_model('animal_inception_fine_tuned.h5')

for i in range(0, len(input_images)) :
    img_path = input_images[i].rstrip()
    img = image.load_img("train/" + img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = np.expand_dims(pred, axis=0)
    if i == 0 :
        out_matrix = pred
    else :
        out_matrix = np.append(pred,axis=0)
    
np.savetxt('prediction.csv', out_matrix)