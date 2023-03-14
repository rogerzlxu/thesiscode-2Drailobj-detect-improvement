import glob
import numpy as np
import pandas as pd
import os
import shutil

# make customized kitti format dataset
# For example: I want all images with labels in folder 0803 and 0807 as training set
#              then I want all images in folder 0759 as testing set
# The idea is: for the training set, if it has 1001 images, then the names should be
#              000000.jpg to 001000.jpg, and if the testing set has 2000 images, the names
#              should be 001001.jpg to 003000.jpg

# set the training & testing path
dir_train = './data/datasets/training_0807+0759/'
dir_test = './data/datasets/testing_0803+0930/'

# processing the training set
# get all the names of image set
nam_image_train = glob.glob(dir_train + 'training/image_2/*.jpg')
# get all the names of labels
nam_label_train = glob.glob(dir_train + 'training/label_2/*.txt')
# get all the names of bins
nam_velo_train = glob.glob(dir_train + 'training/velodyne/*.bin')
# get all the names of calib
name_calib_train = glob.glob(dir_train + 'training/calib/*.txt')

# create a list only contain kitti number of the training files
train_file_num = []
for file in nam_image_train:
    train_file_num.append(file[-10:-4])

# generate new names for the training files
# first generate a list of numbers, such as 0-1000 for the 1000 files
num_new_train = np.arange(0, len(nam_label_train), 1)
# then make lists to store the strings of new names
train_index_new = []
for num in num_new_train:
    train_index_new.append(str(num))
# make a list store strings of new names in kitti format
train_index_new_kitti = []
for index in train_index_new:
    if len(index) == 1:
        train_index_new_kitti.append('00000' + index)
    elif len(index) == 2:
        train_index_new_kitti.append('0000' + index)
    elif len(index) == 3:
        train_index_new_kitti.append('000' + index)
    elif len(index) == 4:
        train_index_new_kitti.append('00' + index)
    elif len(index) == 5:
        train_index_new_kitti.append('0' + index)
    else:
        train_index_new_kitti.append(index)

# create the list of directories to be renamed
train_new_dir_calib = []
train_new_dir_image = []
train_new_dir_label = []
train_new_dir_velodyne = []

for index in train_index_new:
    if len(index) == 1:
        train_new_dir_calib.append(dir_train + 'training/calib/00000' + index + '.txt')
        train_new_dir_image.append(dir_train + 'training/image_2/00000' + index + '.jpg')
        train_new_dir_label.append(dir_train + 'training/label_2/00000' + index + '.txt')
        train_new_dir_velodyne.append(dir_train + 'training/velodyne/00000' + index + '.bin')
    elif len(index) == 2:
        train_new_dir_calib.append(dir_train + 'training/calib/0000' + index + '.txt')
        train_new_dir_image.append(dir_train + 'training/image_2/0000' + index + '.jpg')
        train_new_dir_label.append(dir_train + 'training/label_2/0000' + index + '.txt')
        train_new_dir_velodyne.append(dir_train + 'training/velodyne/0000' + index + '.bin')
    elif len(index) == 3:
        train_new_dir_calib.append(dir_train + 'training/calib/000' + index + '.txt')
        train_new_dir_image.append(dir_train + 'training/image_2/000' + index + '.jpg')
        train_new_dir_label.append(dir_train + 'training/label_2/000' + index + '.txt')
        train_new_dir_velodyne.append(dir_train + 'training/velodyne/000' + index + '.bin')
    elif len(index) == 4:
        train_new_dir_calib.append(dir_train + 'training/calib/00' + index + '.txt')
        train_new_dir_image.append(dir_train + 'training/image_2/00' + index + '.jpg')
        train_new_dir_label.append(dir_train + 'training/label_2/00' + index + '.txt')
        train_new_dir_velodyne.append(dir_train + 'training/velodyne/00' + index + '.bin')
    elif len(index) == 5:
        train_new_dir_calib.append(dir_train + 'training/calib/0' + index + '.txt')
        train_new_dir_image.append(dir_train + 'training/image_2/0' + index + '.jpg')
        train_new_dir_label.append(dir_train + 'training/label_2/0' + index + '.txt')
        train_new_dir_velodyne.append(dir_train + 'training/velodyne/0' + index + '.bin')
    else:
        train_new_dir_calib.append(dir_train + 'training/calib/' + index + '.txt')
        train_new_dir_image.append(dir_train + 'training/image_2/' + index + '.jpg')
        train_new_dir_label.append(dir_train + 'training/label_2/' + index + '.txt')
        train_new_dir_velodyne.append(dir_train + 'training/velodyne/' + index + '.bin')

# deal with the flag folder in training folder
# First get all the flag files
train_flags = glob.glob(dir_train + 'flags/*.txt')
# delete the last dir of 'fugro-to-kitti-ids.txt', that will not be counted
del train_flags[-1]
# create a list only contain kitti number of the flags
train_flag_num = []
for flag in train_flags:
    train_flag_num.append(flag[-10:-4])
# check whether the number of flag files is the same as training labels
if len(train_flag_num) != len(train_file_num):
    # if not, find the different flags
    train_flag_diff_names = [num for num in train_flag_num if num not in train_file_num]

# Then delete the different flags
for name in train_flag_diff_names:
    os.remove(dir_train + 'flags/' + name + '.txt')

# Then in the 'fugro-to-kitti-ids.txt', only keep the frames with same names as the training files
# train_kitti_ids = pd.read_csv(dir_train + 'flags/fugro-to-kitti-ids.txt', sep=' ', header=None)
train_kitti_ids = open(dir_train + 'flags/fugro-to-kitti-ids.txt', 'r')
n = len(train_kitti_ids.readlines())
# create a list for all flag ids in kitti format string
train_kitti_ids_list = []
for i in range(n):
    if len(str(i)) == 1:
        train_kitti_ids_list.append('00000' + str(i))
    elif len(str(i)) == 2:
        train_kitti_ids_list.append('0000' + str(i))
    elif len(str(i)) == 3:
        train_kitti_ids_list.append('000' + str(i))
    elif len(str(i)) == 4:
        train_kitti_ids_list.append('00' + str(i))
    elif len(str(i)) == 5:
        train_kitti_ids_list.append('0' + str(i))
    else:
        train_kitti_ids_list.append(str(i))

# get the ids not in training files
train_kitti_ids_diff = [id for id in train_kitti_ids_list if id not in train_file_num]
# get all the lines
train_kitti_ids = open(dir_train + 'flags/fugro-to-kitti-ids.txt', 'r')
train_kitti_ids_lines = train_kitti_ids.readlines()
train_kitti_ids.close()
# create new list only store flags in training set
train_kitti_ids_lines_new = [line for line in train_kitti_ids_lines if line[0:6] not in train_kitti_ids_diff]
for l in range(len(train_kitti_ids_lines_new)):
    train_kitti_ids_lines_new[l] = train_index_new_kitti[l] + train_kitti_ids_lines_new[l][6:]

# now the new ids txt file has been created, let's store it
train_kitti_ids = open(dir_train + 'flags/fugro-to-kitti-ids.txt', "w+")
for line in train_kitti_ids_lines_new:
    train_kitti_ids.write(line)
train_kitti_ids.close()
print(dir_train + 'flags/fugro-to-kitti-ids.txt has been successfully rewrite')
# now the 'fugro-to-kitti-ids.txt' for training set has been successfully rewrite :)

# later, also change the name of the flags
for f in range(len(train_file_num)):
    os.rename(dir_train + 'flags/' + train_file_num[f] + '.txt', dir_train + 'flags/' + train_index_new[f] + '.txt')
    print(dir_train + 'flags/' + train_file_num[f] + '.txt rename to' + dir_train + 'flags/' + train_index_new[f] + '.txt')

for f in range(len(train_file_num)):
    os.rename(dir_train + 'flags/' + train_index_new[f] + '.txt', dir_train + 'flags/' + train_index_new_kitti[f] + '.txt')
    print(dir_train + 'flags/' + train_index_new[f] + '.txt rename to' + dir_train + 'flags/' + train_index_new_kitti[f] + '.txt')

# Then modify the Imageset folder
# first modify 'trainval.txt'
print('start modifying' + dir_train + 'ImageSets/trainval.txt')
train_txt_trainval = open(dir_train + 'ImageSets/trainval.txt', 'r')
train_txt_trainval_lines = train_txt_trainval.readlines()
train_txt_trainval.close()
# rewrite the lines with new names
for l in range(len(train_txt_trainval_lines)):
    train_txt_trainval_lines[l] = train_index_new_kitti[l] + train_txt_trainval_lines[l][6:]
# let's store the new 'trainval.txt'
train_txt_trainval = open(dir_train + 'ImageSets/trainval.txt', 'w+')
for line in train_txt_trainval_lines:
    train_txt_trainval.write(line)
train_txt_trainval.close()
print(dir_train + 'ImageSets/trainval.txt has been successfully renamed')

# Then modify 'val.txt'
# get all the elements and store them in a list (not include the '\n')
print('start modifying' + dir_train + 'ImageSets/val.txt')
train_txt_val = open(dir_train + 'ImageSets/val.txt', 'r')
train_txt_val_lines = train_txt_val.readlines()
train_txt_val.close()
train_txt_val_list = [line[0:6] for line in train_txt_val_lines]
# get the indexes of ids in 'val.txt' in all the training files
train_txt_val_indexes = [train_file_num.index(id) for id in train_txt_val_list]
# then get the new ids for ids in 'val.txt'
train_txt_val_list_new = [train_index_new_kitti[index] for index in train_txt_val_indexes]
# then rewrite the lines of val.txt (add '\n' for each line)
for l in range(len(train_txt_val_lines)):
    train_txt_val_list_new[l] = train_txt_val_list_new[l] + train_txt_val_lines[l][6:]
# let's store the new 'val.txt' file
train_txt_val = open(dir_train + 'ImageSets/val.txt', 'w+')
for line in train_txt_val_list_new:
    train_txt_val.write(line)
train_txt_val.close()
print(dir_train + 'ImageSets/val.txt has been successfully renamed')

# Then modify 'train.txt'
print('start modifying' + dir_train + 'ImageSets/train.txt')
train_txt_train = open(dir_train + 'ImageSets/train.txt', 'r')
train_txt_train_lines = train_txt_train.readlines()
train_txt_train.close()
train_txt_train_list = [line[0:6] for line in train_txt_train_lines]
# get the indexes of ids in 'train.txt' in all the training files
train_txt_train_indexes = [train_file_num.index(id) for id in train_txt_train_list]
# then get the new ids for ids in 'train.txt'
train_txt_train_list_new = [train_index_new_kitti[index] for index in train_txt_train_indexes]
# then rewrite the lines of train.txt (add '\n' for each line)
for l in range(len(train_txt_train_lines)):
    train_txt_train_list_new[l] = train_txt_train_list_new[l] + train_txt_train_lines[l][6:]
# let's store the new 'train.txt' file
train_txt_train = open(dir_train + 'ImageSets/train.txt', 'w+')
for line in train_txt_train_list_new:
    train_txt_train.write(line)
train_txt_train.close()
print(dir_train + 'ImageSets/train.txt has been successfully renamed')
# remove test.txt in the training set
os.remove(dir_train + 'ImageSets/test.txt')

# Then rename all the files in training folder
for i in range(len(nam_image_train)):
    # To avoid confliction, first rename name as shorter names
    os.rename(name_calib_train[i], dir_train + 'training/calib/' + train_index_new[i] + '.txt')
    os.rename(nam_image_train[i], dir_train + 'training/image_2/' + train_index_new[i] + '.jpg')
    os.rename(nam_label_train[i], dir_train + 'training/label_2/' + train_index_new[i] + '.txt')
    os.rename(nam_velo_train[i], dir_train + 'training/velodyne/' + train_index_new[i] + '.bin')

for i in range(len(nam_image_train)):
    # Then give them new kitti format name
    os.rename(dir_train + 'training/calib/' + train_index_new[i] + '.txt', dir_train + 'training/calib/' + train_index_new_kitti[i] + '.txt')
    os.rename(dir_train + 'training/image_2/' + train_index_new[i] + '.jpg', dir_train + 'training/image_2/' + train_index_new_kitti[i] + '.jpg')
    os.rename(dir_train + 'training/label_2/' + train_index_new[i] + '.txt', dir_train + 'training/label_2/' + train_index_new_kitti[i] + '.txt')
    os.rename(dir_train + 'training/velodyne/' + train_index_new[i] + '.bin', dir_train + 'training/velodyne/' + train_index_new_kitti[i] + '.bin')


# now the training set has been already successfully rewritten! :)

# Then let's start the modification of the testing set
# processing the testing set
# get all the names of image set
nam_image_test = glob.glob(dir_test + 'testing/image_2/*.jpg')
# get all the names of labels
nam_label_test = glob.glob(dir_test + 'testing/label_2/*.txt')
# get all the names of bins
nam_velo_test = glob.glob(dir_test + 'testing/velodyne/*.bin')
# get all the names of calib
name_calib_test = glob.glob(dir_test + 'testing/calib/*.txt')

# create a list only contain kitti number of the testing files
test_file_num = []
for file in nam_image_test:
    test_file_num.append(file[-10:-4])

# generate new names for the testing files
# first generate a list of numbers, such as 0-1000 for the 1000 files
# the number of testing files should continue with the number of training files
# for example, if number of training files is 0-1765, then testing should start from 1766
num_new_test = np.arange(0, len(nam_label_test), 1)
num_new_test = num_new_test + num_new_train[-1] + 1
# then make lists to store the strings of new names
test_index_new = []
for num in num_new_test:
    test_index_new.append(str(num))
# make a list store strings of new names in kitti format
test_index_new_kitti = []
for index in test_index_new:
    if len(index) == 1:
        test_index_new_kitti.append('00000' + index)
    elif len(index) == 2:
        test_index_new_kitti.append('0000' + index)
    elif len(index) == 3:
        test_index_new_kitti.append('000' + index)
    elif len(index) == 4:
        test_index_new_kitti.append('00' + index)
    elif len(index) == 5:
        test_index_new_kitti.append('0' + index)
    else:
        test_index_new_kitti.append(index)

# deal with the flag folder in testing folder
# First get all the flag files
test_flags = glob.glob(dir_test + 'flags/*.txt')
# delete the last dir of 'fugro-to-kitti-ids.txt', that will not be counted
del test_flags[-1]
# create a list only contain kitti number of the flags
test_flag_num = []
for flag in test_flags:
    test_flag_num.append(flag[-10:-4])
# get the indexes of flag ids in all test files
test_flag_ids = [test_file_num.index(i) for i in test_flag_num]
# use the index to get new names for the flags
test_flag_name_new = [test_index_new_kitti[index] for index in test_flag_ids]
# rename the testing flag files
for f in range(len(test_flag_num)):
    os.rename(test_flags[f], dir_test + 'flags/' + test_index_new[f] + '.txt')
for f in range(len(test_flag_num)):
    os.rename(dir_test + 'flags/' + test_index_new[f] + '.txt', dir_test + 'flags/' + test_flag_name_new[f] + '.txt')

# Then in the 'fugro-to-kitti-ids.txt', change the ids of all lines
# first open the file and get all the lines
test_flag_kitti_ids = open(dir_test + 'flags/fugro-to-kitti-ids.txt', 'r')
test_flag_kitti_ids_lines = test_flag_kitti_ids.readlines()
test_flag_kitti_ids.close()
# change the ids
for l in range(len(test_flag_kitti_ids_lines)):
    test_flag_kitti_ids_lines[l] = test_index_new_kitti[l] + test_flag_kitti_ids_lines[l][6:]
# store the new 'fugro-to-kitti-ids.txt
test_flag_kitti_ids = open(dir_test + 'flags/fugro-to-kitti-ids.txt', 'w+')
for line in test_flag_kitti_ids_lines:
    test_flag_kitti_ids.write(line)
test_flag_kitti_ids.close()

# Rewrite the files in folder ImageSets
# For the testing folder, we only need to rewrite 'test.txt'
test_txt_test = open(dir_test + 'ImageSets/test.txt', 'r')
test_txt_test_lines = test_txt_test.readlines()
test_txt_test.close()
test_txt_test_list = [line[0:6] for line in test_txt_test_lines]
# get the indexes of ids in 'test.txt' in all the testing files
test_txt_test_indexes = [test_file_num.index(id) for id in test_txt_test_list]
# then get the new ids for ids in 'test.txt'
test_txt_test_list_new = [test_index_new_kitti[index] for index in test_txt_test_indexes]
# then rewrite the lines of test.txt (add '\n' for each line)
for l in range(len(test_txt_test_lines)):
    test_txt_test_list_new[l] = test_txt_test_list_new[l] + test_txt_test_lines[l][6:]
# let's store the new 'test.txt' file
test_txt_test = open(dir_test + 'ImageSets/test.txt', 'w+')
for line in test_txt_test_list_new:
    test_txt_test.write(line)
test_txt_test.close()
# remove other files in ImageSets folder
os.remove(dir_test + 'ImageSets/train.txt')
os.remove(dir_test + 'ImageSets/trainval.txt')
os.remove(dir_test + 'ImageSets/val.txt')

# Rename the files in the testing folder
for i in range(len(nam_image_test)):
    # To avoid confliction, first rename name as shorter names
    os.rename(name_calib_test[i], dir_test + 'testing/calib/' + test_index_new[i] + '.txt')
    os.rename(nam_image_test[i], dir_test + 'testing/image_2/' + test_index_new[i] + '.jpg')
    os.rename(nam_label_test[i], dir_test + 'testing/label_2/' + test_index_new[i] + '.txt')
    os.rename(nam_velo_test[i], dir_test + 'testing/velodyne/' + test_index_new[i] + '.bin')

for i in range(len(nam_image_test)):
    # Then give them new kitti format name
    os.rename(dir_test + 'testing/calib/' + test_index_new[i] + '.txt', dir_test + 'testing/calib/' + test_index_new_kitti[i] + '.txt')
    os.rename(dir_test + 'testing/image_2/' + test_index_new[i] + '.jpg', dir_test + 'testing/image_2/' + test_index_new_kitti[i] + '.jpg')
    os.rename(dir_test + 'testing/label_2/' + test_index_new[i] + '.txt', dir_test + 'testing/label_2/' + test_index_new_kitti[i] + '.txt')
    os.rename(dir_test + 'testing/velodyne/' + test_index_new[i] + '.bin', dir_test + 'testing/velodyne/' + test_index_new_kitti[i] + '.bin')

# Finally, merge the flags folder
train_ftk_ids = open(dir_train + 'flags/fugro-to-kitti-ids.txt', 'a')
test_ftk_ids = open(dir_test + 'flags/fugro-to-kitti-ids.txt', 'r')
test_ftk_ids_lines = test_ftk_ids.readlines()
test_ftk_ids.close()
for line in test_ftk_ids_lines:
    train_ftk_ids.write(line)
train_ftk_ids.close()
# move all flags in test set to train set
test_flags = glob.glob(dir_test + 'flags/*.txt')
del test_flags[-1]
for i in range(len(test_flags)):
    shutil.move(test_flags[i], dir_train + 'flags/' + test_flags[i][-10:-4] + '.txt')


