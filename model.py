import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

def dense(input_shape, output_shape):
#     model = keras.models.Sequential()
#     model.add(keras.layers.Dense(256, input_shape=(input_shape[1],), activation='relu'))
#     model.add(keras.layers.Dense(256, activation='relu'))
#     model.add(keras.layers.Dense(512, activation='relu'))
#     model.add(keras.layers.Dense(1024, activation='relu'))
#     model.add(keras.layers.Dense(512, activation='relu'))
#     model.add(keras.layers.Dense(256, activation='relu'))
#     model.add(keras.layers.Dense(128, activation='relu'))
#     model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dense(output_shape[1], activation='softmax'))

    model = keras.models.Sequential()
    model.add(Dense(512, input_dim=input_shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_shape[1]))
    model.add(Activation('softmax'))

    return model


def rnn(input_shape, output_shape):
    model = keras.models.Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape[1:]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_shape[1]))
    model.add(Activation('softmax'))

    return model

def embedding_dense(input_shape, output_shape):
    model = keras.models.Sequential()
    model.add(Dense(512, input_dim=input_shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_shape[1]))
    model.add(Activation('sigmoid'))

def embedding_rnn(input_shape, output_shape):
    model = keras.models.Sequential()
    model.add(Dense(512, input_dim=input_shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_shape[1]))
    model.add(Activation('softmax'))

def build_model(model_type, input_shape, output_shape,):
    if 'rnn' == model_type:
        return rnn(input_shape, output_shape)
    else:
        return dense(input_shape, output_shape)

def build_embed_model(model_type, input_shape, output_shape,):
    if 'rnn' == model_type:
        return embedding_rnn(input_shape, output_shape)
    else:
        return embedding_dense(input_shape, output_shape)