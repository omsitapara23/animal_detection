import numpy as np
import cv2

inp_dm = 224

X_train = np.zeros((13000,inp_dm,inp_dm,3))

for i in range(13000):
    image = cv2.imread('train/Img-{}.jpg'.format(i+1))
    #print(image.shape)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray_image)
#     plt.show()
#     plt.imshow(image)
#     plt.show()
    resized_image = cv2.resize(image, (inp_dm, inp_dm)) 
    X_train[i] =  resized_image
    if i % 100 == 0:
        print(i)

print(X_train)

np.save('trainDataset.npy', X_train)