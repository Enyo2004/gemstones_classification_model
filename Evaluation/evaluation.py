import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to eliminate warning message when running the program
os.environ['KERAS_BACKEND'] = 'torch' # use pytorch for keras backend for GPU Usage (faster training)

from Dataset.GetDataset import * # import everything from the dataset

# set global policy to mixed_float16 for faster training
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# load all the models
import keras # import keras to use load model method

# import the models 
from Models.load_models.loadModels import *

# evaluate with the validation data in each model
evaluation1 = model1.evaluate(validation_data)
evaluation2 = model2.evaluate(validation_data)
evaluation3 = model3.evaluate(validation_data)
evaluation4 = model4.evaluate(validation_data)
evaluation5 = model5.evaluate(validation_data)
evaluation6 = model6.evaluate(validation_data)

# pandas dataframe of the evaluation's accuracy
import pandas as pd
models_accuracy = pd.DataFrame({
    'model1': [evaluation1[1], evaluation1[0]],
    'model2': [evaluation2[1], evaluation2[0]],
    'model3': [evaluation3[1], evaluation3[0]],
    'model4': [evaluation4[1], evaluation4[0]],
    'model5': [evaluation5[1], evaluation5[0]],
    'model6': [evaluation6[1], evaluation6[0]],

}, index=['accuracy', 'cross_entropy']).sort_values(by=['accuracy'], axis=1, ascending=False) # sort the dataframe by accuracy (in descending order)

print(models_accuracy) # see the pandas dataFrame to compare best models 


# plot the dataFrame for data visualization of the models performance
models_accuracy.plot(kind='bar', rot=0) # rotate the letters so the whole metric is seen

from matplotlib import pyplot as plt # import matplotlib to use the show method (to plot the bar chart)
plt.show() # show the data


