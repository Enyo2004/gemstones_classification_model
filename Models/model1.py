import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to eliminate warning message when running the program
os.environ['KERAS_BACKEND'] = 'torch' # use pytorch for keras backend for GPU Usage (faster training)

from Dataset.GetDataset import * # import everything from the dataset

# set global policy to mixed_float16 for faster training
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# create the model (provide the layers)

# data augmentation layer (used to avoid overfitting the model)
from keras import Sequential, layers

# Model with Dense layers
DenseLayersModel = Sequential([
    # input layer
    layers.Input(shape=(IMAGE_SIZE + (3, ))), # shape of the image (BATCH_SIZE, HEIGHT, WIDTH, COLOR_CHANNELS)

    # rescale the images (for better accuracy)
    layers.Rescaling(scale=1./255.), # normalize the images

    # hidden layers (to model non-linear data)
    layers.Dense(20, activation='relu'),

    layers.Dense(20, activation='relu'),

    layers.GlobalAveragePooling2D(), # Average Pool to get the best features
     
    # output layer (output a probability tensor of each class)
    layers.Dense(len(class_names)), 

    # use of the activation function softmax to turn numbers into probabilities (to make the model more confident in each prediction)
    layers.Activation(activation='softmax', dtype=tf.float32), # activation function with float32 for numerical precision

])


# see model's summary of the layers
DenseLayersModel.summary()


# take a peek at the name, datatype and global policy per layer
for layer in DenseLayersModel.layers:
    print(layer.name, layer.dtype, layer.dtype_policy)



# compile the model (provide the loss metrics and the optimizer function for the backpropagation)
from keras import losses, optimizers
DenseLayersModel.compile(
    loss=losses.sparse_categorical_crossentropy, # use of sparse categorical crossentropy due to number encoded and multi-class classification (more than 2 classes)
    optimizer=optimizers.Adam(learning_rate=0.001), # use of Adam (adaptive estimator), due to having great results in images classification 
    metrics=['accuracy'] # accuracy as metric to see how accurate predictions are in unseen data
)


# provide callbacks (extra add-ons to the model)
from keras.callbacks import ModelCheckpoint, EarlyStopping

stop = EarlyStopping( # used to stop when the model starts having worse results in the following 'patience' epochs
    monitor='val_accuracy', # to see if the model starts overfitting or underfitting
    patience=3, # 3 epochs of worse results to stop training
    verbose=0 # don't show the early stopping process
)


# train the model 
DenseModelHistory = DenseLayersModel.fit(
    train_data, # fit the train data
    epochs=10, # fit for 10 epochs
    steps_per_epoch=len(train_data), # train per batch (32)

    validation_data=test_data, # validate with the test data
    validation_steps=len(test_data), # validate per batch (32)

    callbacks=[stop] # provide the callback (extra add-on)
)



# evaluate the model (evaluate with the validation data)
evaluation = DenseLayersModel.evaluate(validation_data)
print(f"Loss: {evaluation[0]} | Accuracy: {evaluation[1]}") # print the loss and the accuracy

# save the model 
DenseLayersModel.save('Models/models/model1/DenseModel.keras')

# save the weights learned by the model
DenseLayersModel.save_weights('Models/models/model1/DenseModel.weights.h5')


from Functions.helperFunctions import * # import the helper Functions to plot the loss curves (see if model is overfitting, underfitting or performing well)
plot_loss_curves(DenseModelHistory) # plot loss curves
plt.show() # show the plot















