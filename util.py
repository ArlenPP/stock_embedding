import pandas as pd
import numpy as np
from sklearn import manifold, datasets

def plot_roi(roi, result, plot=False):
    roi_df = pd.DataFrame(roi, index=result.Date, columns=['ROI'])
    if plot: fig = roi_df.plot(rot=75)
    return roi_df


def categori_reverse(pred):
    temp_z = np.argmax(pred, axis=1)
    pred_z = []
    for z in temp_z:
        if 2 == z: pred_z.append(-1)
        else: pred_z.append(z)
    return pred_z

def tsne_trans(input):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    return tsne.fit_transform(input)