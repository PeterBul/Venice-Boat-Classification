from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from scipy.misc import imread
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

TRAIN = True
# image dimensions
IMG_HEIGHT, IMG_WIDTH =  240, 800
# Change number of epochs from time available. One epoch takes aprox. 1 minute
EPOCHS = 50
BATCH_SIZE = 32

# Paths to the different datasets
TRAIN_DATA = r'data/generators/train'
VALID_DATA = r'data/generators/valid'
TEST_DATA = r'data/generators/test/'
SAVE_WEIGHTS_PATH = 'output/vbc/vbc_5.h5'
LOAD_WEIGHTS_PATH = 'output/vbc/third_try.h5'

# Class weights have to correspond to the number of categories. The weights are
# used to handle unbalanced datasets. The numbers should be approximately the
# ration of the sizes of the datasets.
CLASS_WEIGHT = {0: 1.,
                1: 1.,
                2: 2.}

def main():
    boats = os.listdir(TRAIN_DATA)
    model = prepare_model(boats)
    if TRAIN:
        train_generator, validation_generator = prepare_generators()
        model = fit_model(model, train_generator, validation_generator)
    else:
        model.load_weights(LOAD_WEIGHTS_PATH)
    model.summary()
    test_model(model)


def prepare_model(boats):
    # Make sure the input shape is handled correctly by checking the format
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_HEIGHT, IMG_WIDTH)
    else:
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    # Build the different layers of the CNN
    model = Sequential()

    # 1st group with a convolutional layer and pooling
    model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 2nd group with a convolutional layer and pooling
    model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 3rd group with a convolutional layer and pooling
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten array to a one dimensional array for use in a fully connected layer
    model.add(Flatten())
    # Fully connected layer with 64 nodes for detectic higher level features
    model.add(Dense(64, activation='relu'))
    # Add dropout layer to avoid overfitting
    model.add(Dropout(0.5))
    # Fully connected 4 node layer with softmax activation because we have
    # 4 categories.
    model.add(Dense(len(boats), activation='softmax'))

    # Compile the model with categorical parameters
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def prepare_generators():
    # Generate augumented images to expand number of samples in the dataset
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rescale=1./255,
            fill_mode='nearest'
    )

    # ImageDataGenerator for validation is only rescaled
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Make generators to generate augumented images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        VALID_DATA,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)
    return train_generator, validation_generator


def fit_model(model, train_generator, validation_generator):
    # Fit CNN model
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = validation_generator.n//validation_generator.batch_size
    model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=step_size_valid,
        class_weight=CLASS_WEIGHT)

    # Save weights for later use
    model.save_weights(SAVE_WEIGHTS_PATH)
    return model


def test_model(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(TEST_DATA,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            shuffle=False)

    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_generator, test_generator.n // test_generator.batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=BOATS))


if __name__ == '__main__':
    main()
