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

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.optimizers import SGD


# create the base pre-trained model
model = ResNet50(weights='imagenet')
model.layers.pop()


for layer in model.layers:
    layer.trainable=False

last = model.layers[-1].output
x = Dense(30, activation="softmax")(last)
finetuned_model = Model(model.input, x)

# # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)

# # let's add a fully-connected layer
# x = Dense(512, activation='relu')(x)

# # and a logistic layer 
# predictions = Dense(30, activation='softmax')(x)



finetuned_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train = np.load('trainDataset.npy')

finetuned_model.fit(x=X_train/255., y=y, batch_size= 64, epochs= 25, validation_split=0.1, verbose=1)

finetuned_model.save('animal_fine_renet50_V0.h5')
    
layer_num = len(finetuned_model.layers)
for layer in finetuned_model.layers[:int(layer_num * 0.9)]:
    layer.trainable = False
for layer in finetuned_model.layers[int(layer_num * 0.9):]:
        layer.trainable = True

# training
finetuned_model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
finetuned_model.fit(x=X_train/255., y=y, batch_size= 64, epochs= 25, validation_split=0.1, verbose=1)

finetuned_model.save('animal_fine_renet50_V1.h5')
    



