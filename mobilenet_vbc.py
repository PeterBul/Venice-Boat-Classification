import os
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from scipy.misc import imread, imresize
from sklearn.metrics import classification_report, confusion_matrix

TRAIN = True
EPOCHS = 50
BATCH_SIZE = 32
IMG_HEIGHT = 240
IMG_WIDTH = 800
TRAIN_DATA = r'data/generators/train'
VALID_DATA = r'data/generators/valid'
TEST_DATA = r'data/generators/test/'
SAVE_WEIGHTS_PATH = r'output/mobilenet/MobileNet_8.h5'
LOAD_WEIGHTS_PATH = r'output/mobilenet/MobileNet_7.h5'
CLASS_WEIGHT = {0: 1.,
                1: 1.,
                2: 2.}

def main():
    boats = list(filter(lambda f: os.path.isdir(os.path.join(TRAIN_DATA, f)),
                        os.listdir(TRAIN_DATA)))
    model = prepare_model(boats)
    if TRAIN:
        train_generator, validation_generator = prepare_generators()
        model = fit_model(model, train_generator, validation_generator)
    else:
        model.load_weights(LOAD_WEIGHTS_PATH)
    model.summary()


def prepare_model(boats):
    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(len(boats), activation='softmax')(x)

    model = Model(inputs=base_model.input,outputs=preds)
    model.summary()

    for layer in model.layers[:-4]:
        layer.trainable=False
    for layer in model.layers[-4:]:
        layer.trainable=True

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def prepare_generators():
    train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input,
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(TRAIN_DATA,
                                                        target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                        color_mode='rgb',
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=True)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_generator = valid_datagen.flow_from_directory(VALID_DATA,
                                                        target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                        color_mode='rgb',
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=False)
    return train_generator, validation_generator


def fit_model(model, train_generator, validation_generator):

    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = validation_generator.n//validation_generator.batch_size
    model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,
                       epochs=EPOCHS,
                       validation_data=validation_generator,
                       validation_steps=step_size_valid,
                       class_weight=CLASS_WEIGHT)
    model.save_weights(SAVE_WEIGHTS_PATH)
    return model


def test_model():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(TEST_DATA,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            shuffle=False)
    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_generator, test_generator.n // test_generator.batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=BOATS))


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def prepare_image(file, crop):
    img_path = ''
    img_array = imread(img_path + file)
    crop_height = (IMG_HEIGHT-224)//2
    crop_width = (IMG_WIDTH-224)//2
    if(crop):
        img_array = img_array[crop_height:crop_height+224, crop_width:crop_width+224]
    else:
        img_array = imresize(img_array, (224,224))

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

if __name__ == '__main__':
    main()
