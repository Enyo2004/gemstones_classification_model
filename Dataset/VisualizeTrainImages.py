from Dataset.GetDataset import *    # import everything from the GetDataset file

# import image visualizer library
from matplotlib import pyplot as plt


# iterator to get the image and label (take one image from first batch only (32 images))
for image, label in train_data.take(1):
    
    plt.figure(figsize=(10,10)) # initialize the figure for the 5 sample images

    total_images = 6 # number of images to plot
    
    # provide 5 sample images with their titles
    for number_image in range(total_images):
        # provide random image from the train dataset (first batch)
        from random import randint
        random_image = randint(0, BATCH_SIZE - 1) # value between 0 and 31 (arrays start at 0)

        plt.subplot(2, total_images//2 , number_image + 1) # subplot to plot the 5 images in a single row
        plt.imshow(image[random_image]/255.) # normalize image so that it can be plotted
        plt.title(class_names[label[random_image].numpy()]) # provide the name of the class in the title of the image
        plt.axis(False) # set axis to False so that there are no ccordinate axis in the image 

plt.show() # show the images       
    















