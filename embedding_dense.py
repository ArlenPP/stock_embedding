from pickle import dump
from data.gen_data import build_X, build_Y, build_XY
from data.gen_data_config import gen_data_config
from evaluation import score
import pandas as pd
import numpy as np
from util import plot_roi, categori_reverse, tsne_trans
from model import build_model, build_embed_model
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sys import argv

embedding_shape = int(argv[1])
batch_size = 128
epochs = 300

data_s, data_e = '2001/1/1', '2018/12/31'
target_s, target_e = '2019/1/1', '2019/12/31'
feature_days = 3

sne_dict = {
    'data':{},
    'tar':{}
}
embed_dict = {
    'data':{},
    'tar':{}
}

if __name__ == "__main__":
    x, y, data_result = build_XY(data_s, data_e, 
                                gen_data_config['selected_features'], 
                                feature_days)
    tar_x, tar_y, tar_result = build_XY(target_s, target_e, 
                                gen_data_config['selected_features'], 
                                feature_days)
    '''for t-SNE'''
    feature_day = 1

    sne_data, _, _ = build_XY(data_s, data_e, 
                                    ['Open', 'Close'], 
                                    feature_day)
    sne_tar, _, _ = build_XY(target_s, target_e, 
                                    ['Open', 'Close'], 
                                    feature_day)
    sne_tr_y = []
    sne_te_y = []
    # data
    for r in sne_data:
        if(r[1] > r[0]):
            sne_tr_y.append(0)
        else:
            sne_tr_y.append(1)
    # target
    for r in sne_tar:
        if(r[1] > r[0]):
            sne_te_y.append(0)
        else:
            sne_te_y.append(1)

    task = '1'
    '''Data'''
    locals()[f'task_{task}_y'] = x[:, len(gen_data_config['selected_features']):]

    '''build model'''
    model = build_embed_model('dense', 
                            x.shape, 
                            locals()[f'task_{task}_y'].shape, 
                            embedding_shape, 
                            'sigmoid')
    sgd = SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error',
                    optimizer=sgd,
                    metrics=['accuracy'])

    '''Fit models and use validation_split=0.1 '''
    history = model.fit(x, 
                        locals()[f'task_{task}_y'],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        shuffle=True,
                        validation_split=0.1)

    '''get embedding output'''
    get_layer_output = keras.backend.function([model.layers[0].input],
                                    [model.layers[5].output])
    embed_dict['data'][task] = get_layer_output([x])[0]
    embed_dict['tar'][task] = get_layer_output([tar_x])[0]

    '''data t-SNE'''
    sne_dict['data'][task]  = tsne_trans(embed_dict['data'][task])
    sne_dict['tar'][task] = tsne_trans(embed_dict['tar'][task])

    for task in ['2a', '2b']:
        if '2a' == task: feature_day = 0
        else: feature_day = 1
        
        task_2, _, _ = build_XY(data_s, data_e, 
                                        ['Open', 'Close'], 
                                    feature_day)

        task_2_y = []
        for r in task_2:
            if(r[1] > r[0]):
                task_2_y.append(0)
            else:
                task_2_y.append(1)
        task_2_y = keras.utils.to_categorical(task_2_y, 2)
        '''build model'''
        model = build_embed_model('dense', x.shape, task_2_y.shape, 
                                        embedding_shape, 'softmax')
        model.compile(loss='binary_crossentropy',
                        optimizer='Adam',
                        metrics=['accuracy'])

        '''Fit models and use validation_split=0.1 '''
        history = model.fit(x, 
                            task_2_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            shuffle=True,
                            validation_split=0.1)

        '''get embedding output'''
        get_layer_output = keras.backend.function([model.layers[0].input],
                                        [model.layers[5].output])
        embed_dict['data'][task] = get_layer_output([x])[0]
        embed_dict['tar'][task] = get_layer_output([tar_x])[0]

        '''data t-SNE'''
        sne_dict['data'][task]  = tsne_trans(embed_dict['data'][task])
        sne_dict['tar'][task] = tsne_trans(embed_dict['tar'][task])

    '''t-SNE'''
    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)

    for i, t in zip(range(1,4), ['1', '2a', '2b']):
        i = i*2
        left, right = i-1, i
        '''plot left'''
        X_tsne = sne_dict['data'][t]
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 正規化
        for i in range(X_norm.shape[0]):
            locals()[f'ax{left}'].text(X_norm[i, 0], X_norm[i, 1], str(sne_tr_y[i]), color=plt.cm.Set1(sne_tr_y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
        '''plot right'''
        X_tsne = sne_dict['tar'][t]
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 正規化
        for i in range(X_norm.shape[0]):
            locals()[f'ax{right}'].text(X_norm[i, 0], X_norm[i, 1], str(sne_te_y[i]), color=plt.cm.Set1(sne_te_y[i]), 
                fontdict={'weight': 'bold', 'size': 9})

        locals()[f'ax{left}'].set_title('train', fontsize=20)
        locals()[f'ax{right}'].set_title('test', fontsize=20)

    fig.suptitle(f't-SNE model:LSTM_{embedding_shape}', fontsize=30)

    fig.savefig(f'./fig/tsne_dense_{embedding_shape}.png')
    file = open(f'./embedding_data/dense_{embedding_shape}', 'wb')
    dump(embed_dict, file)