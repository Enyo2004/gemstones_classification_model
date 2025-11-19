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
from keras.applications import EfficientNetB0 # import efficientNet

# import the efficient net and assign it to a variable
base_model = EfficientNetB0(include_top=False) # do not include the top layers due that the output is from 1000 classes 

# freeze the layers (make them untrainable for feature extraction)
base_model.trainable = False


# Create the feature extraction model (use the functional api for this)
inputs = layers.Input(shape=((IMAGE_SIZE) + (3,))) # input layer

augmentation_layer = augmentation(inputs) # pass the inputs to augmentation layer

efficient_layer = base_model(augmentation_layer, training=False) # pass the augmentation layer to efficient Net

globalMaxPooling_layer = layers.GlobalAveragePooling2D()(efficient_layer) # pass the efficient Net layer to the global avg pooling to extract most important features

dense_layer = layers.Dense(len(class_names))(globalMaxPooling_layer) # pass the global Average pooling to the dense layer where it predicts for each class

outputs = layers.Activation(activation='softmax', dtype=tf.float32)(dense_layer) # pass the dense to the activation function to convert numbers into probabilites and use float32 for numerical precision

# join the layer to create the model
from keras import Model
featureExtractionAugModel = Model(inputs, outputs)

# check model's summary 
featureExtractionAugModel.summary()


# take a peek at the name, datatype and global policy per layer
for layer in featureExtractionAugModel.layers:
    print(layer.name, layer.dtype, layer.dtype_policy)



# compile the model (provide the loss metrics and the optimizer function for the backpropagation)
from keras import losses, optimizers
featureExtractionAugModel.compile(
    loss=losses.sparse_categorical_crossentropy, # use of sparse categorical crossentropy due to number encoded and multi-class classification (more than 2 classes)
    optimizer=optimizers.Adam(learning_rate=0.001), # use of Adam (adaptive estimator), due to having great results in images classification 
    metrics=['accuracy'] # accuracy as metric to see how accurate predictions are in unseen data
)

# provide callbacks (extra add-ons to the model)
from keras.callbacks import ModelCheckpoint, EarlyStopping

stop = EarlyStopping( # used to stop when the model starts having worse results in the following 'patience' epochs
    monitor='val_accuracy', # to see if the model starts overfitting or underfitting
    patience=2, # 2 epochs of worse results to stop training
    verbose=0 # don't show the early stopping process
)


# train the model 
featureExtractionAugModel_history = featureExtractionAugModel.fit(
    train_data, # fit the train data
    epochs=10, # fit for 10 epochs
    steps_per_epoch=int(0.20 * len(train_data)), # train per batch (32) on 10% of data

    validation_data=test_data, # validate with the test data
    validation_steps=int(0.20 * len(test_data)), # validate per batch (32)  on 10% of data

    callbacks=[stop] # provide the callback (extra add-on)
)


# evaluate the model (evaluate with the validation data)
evaluation = featureExtractionAugModel.evaluate(validation_data)
print(f"Loss: {evaluation[0]} | Accuracy: {evaluation[1]}") # print the loss and the accuracy


from Functions.helperFunctions import * # import the helper Functions to plot the loss curves (see if model is overfitting, underfitting or performing well)
plot_loss_curves(featureExtractionAugModel_history) # plot loss curves
plt.show() # show the plot




