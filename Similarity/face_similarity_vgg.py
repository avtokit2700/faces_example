from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define VGG architecture
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('vgg_face_weights.h5')


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


epsilon = 0.40  # cosine similarity


# epsilon = 120 #euclidean distance

def verifyFace(img1, img2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]
    cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    if cosine_similarity < epsilon:
        answer = "verified... they are same person"
    else:
        answer = "unverified! they are not same person!"
    return answer


start_time = time.time()
path_1 = r'vlad.jpg'
path_2 = r'vlad2.jpg'
answer = verifyFace(path_1, path_2)
print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(figsize=(7, 3))

plt.subplot(221)
plt.imshow(image.load_img(path_1, target_size=(224, 224)))
plt.title('1', size=20)
plt.axis('off')
plt.subplot(222)
plt.imshow(image.load_img(path_2, target_size=(224, 224)))
plt.title('2', size=20)
plt.axis('off')
plt.gcf().text(0.5, 0.3, answer, size=18, ha="center", va="center",
               bbox=dict(boxstyle="round",
                         ec=(1., 0.5, 0.5),
                         fc=(1., 0.8, 0.8),
                         )
               )
plt.show()
