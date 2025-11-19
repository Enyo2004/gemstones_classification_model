# import the library of keras to import the already trained models
import keras 

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

