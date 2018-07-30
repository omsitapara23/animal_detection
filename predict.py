import numpy as np
import pandas as pd
from keras.models import load_model

model = load_model('animal_fine_renet50_V1.h5')

data_classes = ["antelope","bat","beaver","bobcat","buffalo","chihuahua","chimpanzee","collie","dalmatian","german+shepherd","grizzly+bear","hippopotamus","horse","killer+whale","mole","moose","mouse","otter","ox","persian+cat","raccoon","rat","rhinoceros","seal","siamese+cat","spider+monkey","squirrel","walrus","weasel","wolf"]


X_test = np.load('testDataset.npy')

print('dataset loaded')

y_test_predict = model.predict(X_test, verbose=1)

label_df = pd.DataFrame(data=y_test_predict, columns= data_classes)

label_df.head(10)

subm = pd.DataFrame()


te_label = pd.read_csv('meta-data/test.csv')


print(te_label['Image_id'])

subm['image_id'] = te_label['Image_id']

print(subm.head(10))
subm = pd.concat([subm, label_df], axis=1)

subm.to_csv('submitvgg.csv',index = False)

print(subm)
    
