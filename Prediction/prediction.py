import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to eliminate warning message when running the program
os.environ['KERAS_BACKEND'] = 'torch' # use pytorch for keras backend for GPU Usage (faster training)

from Dataset.GetDataset import * # import everything from the dataset


# load all the models
import keras # import keras to use load model method

# load model 1 
model1 = keras.models.load_model('Models/models/model1/DenseModel.keras')
# load model 1 weights
model1.load_weights('Models/models/model1/DenseModel.weights.h5')

# load model 2 
model2 = keras.models.load_model('Models/models/model2/ConvolutionalModel.keras')
# load model 2 weights
model2.load_weights('Models/models/model2/ConvolutionalModel.weights.h5')


# load model 3 
model3 = keras.models.load_model('Models/models/model3/FeatureExtractionModel.keras')
# load model 3 weights
model3.load_weights('Models/models/model3/FeatureExtractionModel.weights.h5')


# load model 4 
model4 = keras.models.load_model('Models/models/model4/featureExtractionAugModel.keras')
# load model 4 weights
model4.load_weights('Models/models/model4/featureExtractionAugModel.weights.h5')


# load model 5
model5 = keras.models.load_model('Models/models/model5/fine_tuning_model.keras')
# load model 5 weights
model5.load_weights('Models/models/model5/fine_tuning_model.weights.h5')


# load model 6
model6 = keras.models.load_model('Models/models/model6/EfficientNetB1Model.keras')
# load model 6 weights
model6.load_weights('Models/models/model6/EfficientNetB1Model.weights.h5')


# predict with 6 images from the validation dataset (2 image per class)
from Functions.helperFunctions import pred_and_plot

# get the list of the images in prediction_images folder 
prediction_images = os.listdir('Prediction/prediction_images')

# import matplotlib.pyplot to use some methods from the library
from matplotlib import pyplot as plt


# predict with model 1
# make the size of the figure bigger for the images
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title("Predictions Model 1") # provide a title for the figure

for number_image in range(6):

    # make subplot to plot the 3 different classes in a different row
    plt.subplot(3, 2, number_image + 1)

    # predict and plot
    pred_and_plot(model=model1, filename='Prediction/prediction_images/' + prediction_images[number_image], class_names=class_names)

plt.show()


# predict with model 2
# make the size of the figure bigger for the images
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title("Predictions Model 2") # provide a title for the figure

for number_image in range(6):

    # make subplot to plot the 3 different classes in a different row
    plt.subplot(3, 2, number_image + 1)

    # predict and plot
    pred_and_plot(model=model2, filename='Prediction/prediction_images/' + prediction_images[number_image], class_names=class_names)

plt.show()



# predict with model 3
# make the size of the figure bigger for the images
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title("Predictions Model 3") # provide a title for the figure

for number_image in range(6):

    # make subplot to plot the 3 different classes in a different row
    plt.subplot(3, 2, number_image + 1)

    # predict and plot
    pred_and_plot(model=model3, filename='Prediction/prediction_images/' + prediction_images[number_image], class_names=class_names)

plt.show()


# predict with model 4
# make the size of the figure bigger for the images
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title("Predictions Model 4") # provide a title for the figure

for number_image in range(6):

    # make subplot to plot the 3 different classes in a different row
    plt.subplot(3, 2, number_image + 1)

    # predict and plot
    pred_and_plot(model=model4, filename='Prediction/prediction_images/' + prediction_images[number_image], class_names=class_names)

plt.show()


# predict with model 5
# make the size of the figure bigger for the images
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title("Predictions Model 5") # provide a title for the figure

for number_image in range(6):

    # make subplot to plot the 3 different classes in a different row
    plt.subplot(3, 2, number_image + 1)

    # predict and plot
    pred_and_plot(model=model5, filename='Prediction/prediction_images/' + prediction_images[number_image], class_names=class_names)

plt.show()


# predict with model 6
# make the size of the figure bigger for the images
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title("Predictions Model 6") # provide a title for the figure

for number_image in range(6):

    # make subplot to plot the 3 different classes in a different row
    plt.subplot(3, 2, number_image + 1)

    # predict and plot
    pred_and_plot(model=model6, filename='Prediction/prediction_images/' + prediction_images[number_image], class_names=class_names)

plt.show()

