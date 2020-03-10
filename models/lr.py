import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from DisasterTweets.log import get_logger
from DisasterTweets.utility import LOCATION
logger = get_logger('lr')

def run():
    logger.log('running the lr algorithm')

def get_datasets(filepath, train_ratio = 0.80):
    data = pd.read_csv(filepath)
    train_data = data.sample(frac = train_ratio)
    heldout_data = data.loc[~data.index.isin(train_data.index)]
    return train_data, heldout_data

train_df, heldout_df = get_datasets(LOCATION+'/data/train.csv')
print(train_df.head(5))
