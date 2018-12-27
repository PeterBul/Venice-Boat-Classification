# Project Title

Homework 2 Machine Learning Sapienza 2018

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

TensorFlow
Python 3.x
Scikit-learn

### Set up folder structure

The folder structure needs to be compatible with Keras generators.
You will need a test, validation and test folder. All of which have subfolders,
with the files included in folders named after their category, the same way the train folder
should be organized when downloading from the MarDCT website.
I have included a python file, generate_folder_structure.py, which makes a new folder structure
from the one downloaded at the MarDCT website. This copies all the files to a new folder structure.
It uses two functions: make_train_and_valid_folders() and make_test_folder().
If you would rather make the train and valid folders yourself, feel free to change the
main function, only to use the make_test_folder.

You can also choose the ratio of test and validation set size with TRAIN_SHARE.

After the generating of files, the generators works by reading all the subfolders.
For a quicker demonstration, most folders should be deleted, except the few categories you want to train on. The same categories should be left in train, valid and test folders.

I have trained my system with the 3 first categories in alphabetic order.

Either way, the file structure should be:

```
data/
  train/
    category_1/
      image_1.jpg
      image_2.jpg
      ...
      image_k.jpg
    .
    .
    .
    category_n/
      image_1.jpg
      image_2.jpg
      ...
      image_k.jpg
  valid/
    category_1/
      image_1.jpg
      image_2.jpg
      ...
      image_k.jpg
    .
    .
    .
    category_n/
      image_1.jpg
      image_2.jpg
      ...
      image_k.jpg
  test/
    category_1/
      image_1.jpg
      image_2.jpg
      ...
      image_k.jpg
    .
    .
    .
    category_n/
      image_1.jpg
      image_2.jpg
      ...
      image_k.jpg
```

### Running the files

There are two primary files. vbc.py and mobilenet_vbc.py. They both run on the same folder structure. Vbc is a CNN network built from scratch, mobilenet_vbc.py uses retraining of MobileNet.

To tweak the excecution of the files, the constants in CAPS LOCK should be inspected before running. The number of epochs greatly affects the run time.

## Built With

* [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) - The machine learning frame work used
* [Keras](https://keras.io/) - Machine learning framework from tensorflow


## Authors

* **Peter Cook Bulukin** - *Initial work* - [PeterBul](https://github.com/PeterBul)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
