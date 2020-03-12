'''
    Matt Briones
    Last Modified: Mar10, 2020
'''


import numpy as np
import pandas as pd
from MLutils import separateOutput
from MLutils import gen_rand_data
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from DisasterTweets.log import get_logger
#from DisasterTweets.utility import LOCATION

#logger = get_logger('lr')

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

# TODO: Change to SVM
def learn(train, target, heldout_data, heldout_target):
    ker = 'linear'
    clf = SVC(kernel = ker)
    clf.fit(train, target)
    if(ker == 'linear'):
        print("Weights: ", clf.coef_)
    predictData = clf.predict(heldout_data)
    print("Accuracy Score: ", accuracy_score(predictData, heldout_target))
    return clf

#try using base ML algorithm with randomly generated data
def trySVMWithRand():
    data = gen_rand_data(1000, 5)
    train_df = data.sample(frac = 0.80)
    heldout_df = data.loc[~data.index.isin(train_df.index)]
    train_df, trainOut_df = separateOutput(train_df)
    heldout_df, heldoutOut_df = separateOutput(heldout_df)
    learn(train_df, trainOut_df, heldout_df, heldoutOut_df)

if(testWithRandData):
    trySVMWithRand()
