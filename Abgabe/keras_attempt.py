import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import keras
import tensorflow
from keras.layers import Input, Embedding, Reshape
from keras.models import Model
from keras.applications import ResNet50

# load in data
data = pd.read_csv('data/tokenized/sf_token_training_set.csv')
target_data = data[['rating']]
train_data = data[['tokens']]

for i in target_data:
    if i.isnumeric():
        i = int(i)
        i = round(i)

target_data = target_data.values
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_tokens = encoder.fit_transform(train_data).toarray()

# separate encoded tokens and target rating into separate numpy arrays
encoded_tokens_array = np.array(encoded_tokens)
target_data_array = np.array(target_data)

# create dataset as a tuple of encoded tokens array and target rating array
train_dataset = (encoded_tokens_array, target_data_array)

# define input layer
input_layer = Input(shape=(encoded_tokens_array.shape[1],))

# add embedding layer to convert input to the shape expected by ResNet50
embedded_input = Embedding(input_dim=encoded_tokens_array.shape[1], output_dim=50)(input_layer)
reshaped_input = Reshape((encoded_tokens_array.shape[1], 50, 1))(embedded_input)

# load ResNet50 model
resnet50 = ResNet50(weights=None, include_top=False, input_tensor=reshaped_input)

# add output layer to ResNet50
output = resnet50.output
output = Reshape((-1,))(output)
output = keras.layers.Dense(1, activation='sigmoid')(output)

# create model
model = Model(inputs=input_layer, outputs=output)

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(train_dataset[0], train_dataset[1], batch_size=10, epochs=2)
