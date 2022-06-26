import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MINLOG_LEVEL'] = '3'
def transform_data():
    data = pd.read_csv('./data/predict.csv',header=None)
    data.columns = ['uuid','visit_time','user_id','item_id','features']
    label = pd.read_csv('./data/truth.csv',header=None)
    label.columns = ['uuid','label']
    data = pd.merge(data,label,on='uuid',how='left')
    data.to_pickle('./data/data.pkl')

