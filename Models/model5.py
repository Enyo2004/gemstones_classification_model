import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to eliminate warning message when running the program
os.environ['KERAS_BACKEND'] = 'torch' # use pytorch for keras backend for GPU Usage (faster training)

from Dataset.GetDataset import * # import everything from the dataset

# set global policy to mixed_float16 for faster training
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# import the model 4 history to compare them
from Models.model4 import *

# import the model4 and add it to a new object
import keras
fine_tuning_model = keras.models.load_model('Models/models/model4/featureExtractionAugModel.keras')

# load the weights
fine_tuning_model.load_weights('Models/models/model4/featureExtractionAugModel.weights.h5')

# provide the summary to see if the model was imported
fine_tuning_model.summary()

print(fine_tuning_model.trainable_weights) # print the weights to ensure they were loaded correctly

# unfreeze certain layers to fine tune the model (last 20 layers)
for layer in fine_tuning_model.layers[2].layers[-20:]:
    layer.trainable = True

# provide the summary to see if the layers were unfreezed
fine_tuning_model.summary()

# compile the model
from keras import losses, optimizers
fine_tuning_model.compile(
    loss=losses.sparse_categorical_crossentropy, # use of sparse categorical crossentropy due to number encoded and multi-class classification (more than 2 classes)
    optimizer=optimizers.Adam(learning_rate=0.0001), # lower 10X the learning rate because it is fine tuning
    metrics=['accuracy'] # accuracy as metric to see how accurate predictions are in unseen data
)

# provide callbacks (extra add-ons to the model)
from keras.callbacks import ModelCheckpoint, EarlyStopping

stop = EarlyStopping( # used to stop when the model starts having worse results in the following 'patience' epochs
    monitor='val_accuracy', # to see if the model starts overfitting or underfitting
    patience=5, # 5 epochs of worse results to stop training
    verbose=0 # don't show the early stopping process
)

# fine tune constants
epochs_model4_training = 10 # epochs the model 4 was trained for

total_epochs = epochs_model4_training + 10 # fine tune the model for 10 more epochs

# fine tune the model
fine_tuning_model_history = fine_tuning_model.fit(
    train_data, # training data

    epochs=total_epochs, # total epochs to train 
    initial_epoch=epochs_model4_training, # start from the 7th epoch
    steps_per_epoch=len(train_data), # batch training

    validation_data=test_data, # test the data with unseen data
    validation_steps=len(test_data), # batch testing

    callbacks=[stop] # provide the callback (extra add-on)
)


# evaluate the model (evaluate with the validation data)
evaluation = fine_tuning_model.evaluate(validation_data)
print(f"Loss: {evaluation[0]} | Accuracy: {evaluation[1]}") # print the loss and the accuracy


# save the model 
fine_tuning_model.save('Models/models/model5/fine_tuning_model.keras')

# save the weights learned by the model
fine_tuning_model.save_weights('Models/models/model5/fine_tuning_model.weights.h5')


from Functions.helperFunctions import * # import the helper Functions to plot the loss curves (see if model is overfitting, underfitting or performing well)

# plot the historys (before and after fine-tuning) 
# shows the plot in the function (no need to add it in the end)
compare_historys(original_history=featureExtractionAugModel_history,
                 new_history=fine_tuning_model_history,
                 initial_epochs=epochs_model4_training) 




