import os
import math
from shutil import copyfile

TRAIN_DATA = 'data/sc5'
TEST_FOLDER = 'data/sc5-test'
GROUND_TRUTH = 'data/sc5-test/ground_truth.txt'
TRAIN_SHARE = 0.8

new_generator_folder = 'data/generators'
new_train_folder = os.path.join(new_generator_folder, 'train')
new_valid_folder = os.path.join(new_generator_folder, 'valid')
new_test_folder = os.path.join(new_generator_folder, 'test')

def main():
    make_train_and_valid_folders()
    make_test_folder()

def make_train_and_valid_folders():
    os.mkdir(new_generator_folder)
    os.mkdir(new_train_folder)
    os.mkdir(new_valid_folder)
    os.mkdir(new_test_folder)

    ground_truth_f = open(GROUND_TRUTH)
    ground_truth = {}

    for category_folder in os.listdir(TRAIN_DATA):
        category_folder_path = os.path.join(TRAIN_DATA, category_folder)
        if(os.path.isdir(category_folder_path)):
            len_dir = len(os.listdir(category_folder_path))
            nb_train_files = math.floor(len_dir * TRAIN_SHARE)
            new_category_folder_path = os.path.join(new_train_folder, category_folder)
            new_valid_category_folder_path = os.path.join(new_valid_folder, category_folder)
            os.mkdir(new_category_folder_path)
            os.mkdir(new_valid_category_folder_path)
            for i, file in enumerate(os.listdir(category_folder_path)):
                old_file_path = os.path.join(category_folder_path, file)
                if i <= nb_train_files:
                    new_file_path = os.path.join(new_category_folder_path, file)
                else:
                    new_file_path = os.path.join(new_valid_category_folder_path, file)

                copyfile(old_file_path, new_file_path)

def make_test_folder():
    # make ground truth dictionary
    for line in ground_truth_f:
        file_name = line.split(';')[0]
        label = line.split(';')[-1].strip()
        ground_truth[file_name] = label

    for file, label in ground_truth.items():
        old_file_path = os.path.join(TEST_FOLDER, file)
        new_category_path = os.path.join(new_test_folder, label)
        new_file_path = os.path.join(new_category_path, file)
        if os.path.isdir(new_category_path):
            copyfile(old_file_path, new_file_path)
        else:
            os.mkdir(new_category_path)
            copyfile(old_file_path, new_file_path)

if __name__ == '__main__':
    main()
