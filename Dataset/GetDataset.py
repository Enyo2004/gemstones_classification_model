import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to eliminate warning message when running the program

import tensorflow as tf # import tensorflow library

# import the dataset from the files
from keras.preprocessing import image_dataset_from_directory

# get the directories of the unzipped folders
train_dir = 'Gemstones/train'
test_dir = 'Gemstones/test'
val_dir = 'Gemstones/test'

# provide the constants
BATCH_SIZE = 32 #NUMBER OF IMAGES PASSED IN A BATCH FOR FASTER TRAINING AND VALIDATION
IMAGE_SIZE = (224, 224) # SIZE OF ALL THE IMAGES TO TRAIN AND TEST


# provide the data to train the model
train_data = image_dataset_from_directory(directory=train_dir, # the training data directory
                                          batch_size=BATCH_SIZE, # train data through batches
                                          image_size=IMAGE_SIZE, # image size of train images
                                          label_mode='int', # provides how the class labels will be (int means number encoded classes)
                                          shuffle=True) # shuffle the training data for better performance due to randomness


# provide the test data in order to validate the model's accuracy
test_data = image_dataset_from_directory(directory=test_dir, # test data directory
                                         batch_size=BATCH_SIZE,  # test the data through batches
                                         image_size=IMAGE_SIZE, # image size of test images
                                         label_mode='int') # provides how the class labels will be (int means number encoded classes)


# provide the validation data to evaluate model's performance with unseen data after training
validation_data = image_dataset_from_directory(directory=val_dir, # validation data directory
                                         batch_size=BATCH_SIZE,  # validate the data through batches
                                         image_size=IMAGE_SIZE, # image size of validation images
                                         label_mode='int')# provides how the class labels will be (int means number encoded classes)
                                         


# get the class names 
class_names = train_data.class_names




