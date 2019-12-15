from data.gen_data import build_X, build_Y, build_XY
from data.gen_data_config import gen_data_config
from evaluation import score
import pandas as pd
import numpy as np
from util import plot_roi, categori_reverse
from model import build_model
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
import keras


tr_s, tr_e = '2001/1/1', '2018/12/31'
te_s, te_e = '2019/1/1', '2019/12/31'
feature_days = 3

if __name__ == "__main__":
    
    tr_x, tr_y, tr_result = build_XY(tr_s, tr_e, 
                                     gen_data_config['selected_features'], 
                                     feature_days)
    te_x, te_y, te_result = build_XY(te_s, te_e, 
                                    gen_data_config['selected_features'], 
                                    feature_days)
    # convert class vectors to binary class matrices
    one_tr_y = keras.utils.to_categorical(tr_y, 3)
    one_te_y = keras.utils.to_categorical(te_y, 3)

    model = build_model('dense' ,tr_x.shape, one_tr_y.shape)

    ''' Compile model with specified loss and optimizer '''
    model.compile(loss= 'categorical_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])

    ''' set the size of mini-batch and number of epochs'''
    batch_size = 128
    epochs = 300

    '''Fit models and use validation_split=0.1 '''
    history = model.fit(tr_x, one_tr_y,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                shuffle=True,
                                validation_split=0.1,)

    pred_y = model.predict(te_x)
    pred_z = categori_reverse(pred_y)
    result_roi = score(pred_z, te_result)
    fig = plot_roi(result_roi, te_result)