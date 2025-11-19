from Dataset.GetDataset import *

# create the augmented layer
from keras import Sequential, layers

augmentation = Sequential([
    layers.RandomFlip(mode='horizontal'), # randomly flip certain images horizontally
    layers.RandomInvert(factor=0.2), # randomly invert the image
    layers.RandomCrop(200, 200),    # randomly crop the image to a certain widht or height
    layers.RandomRotation(factor=0.2), # randomly rotate the image
    layers.RandomContrast(factor=0.2) # randomly contrast the color of the image
], name='augmentation_layer')


# provide a random image from the train dataset
from matplotlib import pyplot as plt
import numpy as np

plt.figure(figsize=(7, 7))
for image, label in train_data.take(1):
    # provide random image from the train dataset (first batch)
        from random import randint
        random_image = randint(0, BATCH_SIZE - 1) # value between 0 and 31 (arrays start at 0)

        plt.subplot(1, 2, 1) # plot the normal image
        normal_image = image[random_image]/255. # normalize image so that it can be plotted
        plt.imshow(normal_image) 
        plt.title('Normal Image') # provide the name of the class in the title of the image
        plt.axis(False) # set axis to False so that there are no ccordinate axis in the image 

        plt.subplot(1, 2, 2) # plot the augmented image
        augmented_image = augmentation(normal_image) # augment the already normalized image
        clipped_image = np.clip(augmented_image, 0, 1) # cut the image into range [0,1] 
        plt.imshow(clipped_image) # plot the transformed values of the image (to avoid warnings)
        plt.title('Augmented Image') # provide the name of the class in the title of the image
        plt.axis(False) # set axis to False so that there are no ccordinate axis in the image 

plt.show()

