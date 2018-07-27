import numpy as np
import pandas as pd

dataset = pd.read_csv('meta-data/train.csv')

y = dataset.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features= [0])
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()

with open("train.txt","r") as f:
    input_images= f.readlines()

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer 
predictions = Dense(30, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

for epoch in range (0,5) :
    for i in range(0, len(input_images)) :
        img_path = input_images[i].rstrip()
        img = image.load_img("train/" + img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        model.fit(x,np.transpose(np.expand_dims(y[i],1)), verbose = 0)
    print("completed epoch")
    
layer_num = len(model.layers)
for layer in model.layers[:279]:
    layer.trainable = False

for layer in model.layers[279:]:
    layer.trainable = True

# training
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
for epoch in range (0,5) :
    for i in range(0, len(input_images)) :
        img_path = input_images[i].rstrip()
        img = image.load_img("train/" + img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        model.fit(x,np.transpose(np.expand_dims(y[i],1)), verbose = 0)
    print("completed epoch with layers of inception")
    
from keras.models import load_model

model.save('animal_inception_fine_tuned.h5')


