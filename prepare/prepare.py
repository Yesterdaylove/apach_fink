# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
def id_encoder(data):
    print("Label_encodering--------------------")
    for col in ['uuid']:
        data = data.drop(col,axis=1)
    for col in ['userid','item_id']:
        encoder = LabelEncoder()
        encoder.fit(data[col])
        np.save(col+'_encoder.npy',encoder.classes_)
        data[col] = encoder.transform(data[col])
    return data



