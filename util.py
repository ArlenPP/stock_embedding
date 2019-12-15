import pandas as pd
import numpy as np

def plot_roi(roi, result):
    roi_df = pd.DataFrame(roi, index=result.Date, columns=['ROI'])
    fig = roi_df.plot(rot=75)
    return fig


def categori_reverse(pred):
    temp_z = np.argmax(pred, axis=1)
    pred_z = []
    for z in temp_z:
        if 2 == z: pred_z.append(-1)
        else: pred_z.append(z)
    return pred_z