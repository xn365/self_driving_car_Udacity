import os
import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg

INPUT_SHAPE = (66, 200, 3)
STEERING_CORRECTION = 0.25
BATCH_SIZE = 64
EPOCHS = 5
KEEP_PROB = 0.4

def preprocess(image):

    ratio = float(INPUT_SHAPE[1]) / float(image.shape[1])
    resized_size = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))
    image = cv2.resize(image, dsize=resized_size)
    crop_top = resized_size[1] - INPUT_SHAPE[0]

    image = image[crop_top:, :, :] 
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def random_translation(image, steering_angle, max_move = 20):

    delta = np.random.uniform(-max_move, max_move)
    steering_angle = steering_angle + (delta / max_move) * STEERING_CORRECTION

    move_mat = np.float32([[1,0,delta],[0,1,0]])
    image = cv2.warpAffine(image, move_mat, (image.shape[1],image.shape[0]))

    return image, steering_angle


def read_image(dataset, steering_angle, random = True):
    #Read center left right image randomly if is training, read center image else.
    #dataset[0] center image path
    #dataset[1] left image path
    #dataset[2] right image path
    if random:
        choice = np.random.choice(3)
        if choice == 1:
            steering_angle = steering_angle + STEERING_CORRECTION
        elif choice == 2:
            steering_angle = steering_angle - STEERING_CORRECTION
        image = mpimg.imread(dataset[choice].strip())
        image = preprocess(image)
        image, steering_angle = augmentation(image, steering_angle)
    else:
        choice = 0
        image = mpimg.imread(dataset[choice].strip())
        image = preprocess(image)

    return image, steering_angle

def augmentation(image, steering_angle):

    image, steering_angle = random_translation(image, steering_angle)

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def batch_generator(image_paths, steering_angles, batch_size, is_training):

    images_batch = np.empty([batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
    steerings_batch = np.empty(batch_size)

    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):

            steering_angle = steering_angles[index]
            image, steering_angle = read_image(image_paths[index], steering_angle, is_training)

            images_batch[i] = image
            steerings_batch[i] = steering_angle
            i += 1
            if i == batch_size:
                yield images_batch, steerings_batch
                break


def build_model():
    """
    Define the network and compiles it
    :return: The keras model
    """
    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=INPUT_SHAPE, output_shape=INPUT_SHAPE))

    model.add(Convolution2D(16, kernel_size=5, activation="relu", strides=(2, 2)))
    model.add(Convolution2D(32, kernel_size=5, activation="relu", strides=(2, 2)))
    model.add(Convolution2D(64, kernel_size=5, activation="relu", strides=(2, 2)))
    model.add(Convolution2D(64, kernel_size=3, activation="relu"))
    model.add(Convolution2D(64, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(KEEP_PROB))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(KEEP_PROB))

    model.add(Dense(64,  activation="relu"))
    model.add(Dropout(KEEP_PROB))

    model.add(Dense(32,  activation="relu"))
    model.add(Dropout(KEEP_PROB))

    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
    return model

def train_model(model, n_epochs, train_data, train_label, valid_data, valid_label):

    model.fit_generator(generator=batch_generator(train_data, train_label, 64, True),
                        steps_per_epoch=500,
                        validation_data=batch_generator(valid_data, valid_label, 64, False),
                        validation_steps=50,
                        epochs=n_epochs,
                        verbose=1)



def get_model(n_epochs, train_data, train_label, valid_data, valid_label):

    model = build_model()
    train_model(model, n_epochs, train_data, train_label, valid_data, valid_label)
    return model

def main():
    """
    Main function
    """
    #Get dataset
    log_dir = "./"
    data_df = pd.read_csv(os.path.join(log_dir, 'driving_log.csv'))
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
    #Get model
    model = get_model(EPOCHS, X_train, y_train, X_valid, y_valid)
    model.save(os.path.join("./models/", 'model.h5'))

if __name__ == '__main__':
    main()
