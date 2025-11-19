import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to eliminate warning message when running the program
os.environ['KERAS_BACKEND'] = 'torch' # use pytorch for keras backend for GPU Usage (faster training)

from Dataset.GetDataset import * # import everything from the dataset

# set global policy to mixed_float16 for faster training
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


# data augmentation layer (used to avoid overfitting the model)
from keras import Sequential, layers

augmentation = Sequential([
    layers.RandomFlip(mode='horizontal'), # randomly flip certain images horizontally
    layers.RandomInvert(factor=0.2), # randomly invert the image
    layers.RandomCrop(200, 200),    # randomly crop the image to a certain widht or height
    layers.RandomRotation(factor=0.2), # randomly rotate the image
    layers.RandomContrast(factor=0.2) # randomly contrast the color of the image
], name='augmentation_layer')


# feature extraction model 
from keras.applications import EfficientNetB1 

# import the EfficientNetB1 and assign it to a variable
base_model = EfficientNetB1(include_top=False) # do not include the top layers due that the output is from 1000 classes 

# freeze the layers (make them untrainable for feature extraction)
base_model.trainable = False

# Create the feature extraction model (use the functional api for this)
inputs = layers.Input(shape=((IMAGE_SIZE) + (3,))) # input layer

augmentation_layer = augmentation(inputs) # pass the inputs to augmentation layer

EfficientNetB1_layer = base_model(augmentation_layer, training=False) # pass the augmentation layer to EfficientNetB1

globalAvgPooling_layer = layers.GlobalAveragePooling2D()(EfficientNetB1_layer) # pass the EfficientNetB1 layer to the global avg pooling to extract most important features

dense_layer = layers.Dense(len(class_names))(globalAvgPooling_layer) # pass the global Average pooling to the EfficientNetB1 layer where it predicts for each class

outputs = layers.Activation(activation='softmax', dtype=tf.float32)(dense_layer) # pass the EfficientNetB1 to the activation function to convert numbers into probabilites and use float32 for numerical precision

# join the layer to create the model
from keras import Model
EfficientNetB1Model = Model(inputs, outputs)

# check model's summary 
EfficientNetB1Model.summary()


# take a peek at the name, datatype and global policy per layer
for layer in EfficientNetB1Model.layers:
    print(layer.name, layer.dtype, layer.dtype_policy)



# compile the model (provide the loss metrics and the optimizer function for the backpropagation)
from keras import losses, optimizers
EfficientNetB1Model.compile(
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
EfficientNetB1Model_history = EfficientNetB1Model.fit(
    train_data, # fit the train data
    epochs=10, # fit for 10 epochs
    steps_per_epoch=len(train_data), # train per batch (32)

    validation_data=test_data, # validate with the test data
    validation_steps=len(test_data), # validate per batch (32)

    callbacks=[stop] # provide the callback (extra add-on)
)


# evaluate the model (evaluate with the validation data)
evaluation = EfficientNetB1Model.evaluate(validation_data)
print(f"Loss: {evaluation[0]} | Accuracy: {evaluation[1]}") # print the loss and the accuracy


# save the model 
EfficientNetB1Model.save('Models/models/model6/EfficientNetB1Model.keras')

# save the weights learned by the model
EfficientNetB1Model.save_weights('Models/models/model6/EfficientNetB1Model.weights.h5')


from Functions.helperFunctions import * # import the helper Functions to plot the loss curves (see if model is overfitting, underfitting or performing well)
plot_loss_curves(EfficientNetB1Model_history) # plot loss curves
plt.show() # show the plot






