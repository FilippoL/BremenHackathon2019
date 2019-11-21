# Imports for Deep Learning
import json
import random
# from numpy import floor
from matplotlib import pyplot as plt
from glob import glob
import cv2
from tensorflow import set_random_seed
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, MaxPool2D
from keras import regularizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# ensure consistency across runs
from numpy.random import seed
seed(1)
set_random_seed(2)

# Imports to view data


def plot_three_samples(letter):
    print("Samples images for letter " + letter)
    base_path = './data/asl_alphabet_train/asl_alphabet_train/'
    img_path = base_path + letter + '/**'
    path_contents = glob(img_path)

    plt.figure(figsize=(16, 16))
    imgs = random.sample(path_contents, 3)
    plt.subplot(131)
    plt.imshow(cv2.imread(imgs[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(imgs[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(imgs[2]))
    return


data_dir = "./data/asl_alphabet_train/asl_alphabet_train"
target_size = (64, 64)
target_dims = (64, 64, 3)  # add channel for RGB
n_classes = 3
val_frac = 0.1
batch_size = 32

data_augmentor = ImageDataGenerator(samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)
json.dump(labels, open("./labels.json", "w"))

my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
# my_model.add(BatchNormalization())
# my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
# my_model.add(BatchNormalization())
# my_model.add(Dropout(0.5))
# my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
# my_model.add(BatchNormalization())
# my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
# my_model.add(BatchNormalization())
# my_model.add(Dropout(0.5))
# my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
# my_model.add(BatchNormalization())
# my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
# my_model.add(BatchNormalization())
# my_model.add(Flatten())
# my_model.add(Dense(512, activation='relu'))
# my_model.add(Dense(n_classes, activation='softmax'))


my_model.add(Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
my_model.add(BatchNormalization())
# my_model.add(Dropout(0.7))
my_model.add(MaxPool2D(pool_size=[3, 3]))
my_model.add(Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'))
my_model.add(BatchNormalization())
# my_model.add(Dropout(0.7))
my_model.add(MaxPool2D(pool_size=[3, 3]))
my_model.add(Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Dropout(0.5))
my_model.add(MaxPool2D(pool_size=[3, 3]))
my_model.add(BatchNormalization())
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
my_model.add(Dense(n_classes, activation='softmax'))


my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

my_model.fit_generator(train_generator, epochs=5, validation_data=val_generator)

my_model.save("model.h5.olu")
