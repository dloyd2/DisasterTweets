'''
    Matt Briones
    Last Modified: Mar10, 2020
    Description: Implements logistic regression using scikit-learn
'''


import numpy as np
import pandas as pd
from MLutils import separateOutput
from MLutils import gen_rand_data
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from DisasterTweets.log import get_logger
from DisasterTweets.utility import LOCATION

logger = get_logger('lr')

testWithRandData = True

def run():
#    logger.log('running the lr algorithm')
    train_df, heldout_df = get_datasets(LOCATION+'/data/train.csv')
    train_df, trainOut_df = separateOutput(train_df)
    heldout_df, heldoutOut_df = separateOutput(heldout_df)
    clf = learn(train_df, trainOut_df, heldout_df, heldoutOut_df)

def get_datasets(filepath, train_ratio = 0.80):
    data = pd.read_csv(filepath)
    train_data = data.sample(frac = train_ratio)
    heldout_data = data.loc[~data.index.isin(train_data.index)]
    return train_data, heldout_data

#try using base ML algorithm with randomly generated data
def tryLRWithRand():
    data = gen_rand_data(1000, 5)
    train_df = data.sample(frac = 0.80)
    heldout_df = data.loc[~data.index.isin(train_df.index)]
    train_df, trainOut_df = separateOutput(train_df)
    heldout_df, heldoutOut_df = separateOutput(heldout_df)
    learn(train_df, trainOut_df, heldout_df, heldoutOut_df)

# TODO: Implement LR algorithm
def learn(train, target, heldout_data, heldout_target):
    clf = SGDClassifier(loss = 'log', max_iter = 1000, verbose = 1)
    clf.fit(train, target)
    print("Weights: ", clf.coef_)
    predictData = clf.predict(heldout_data)
    print("Accuracy Score: ", accuracy_score(predictData, heldout_target))
    return clf

if(testWithRandData):
    tryLRWithRand()
