'''
    Matt Briones
    March 10, 2020
    Description: Implements a random forest algorithm using scikit-learn
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from DisasterTweets.log import get_logger
from DisasterTweets.utility import LOCATION

logger = get_logger('rf')

testWithRandData = False

def run():
    logger.log('running the random forest algorithm')
    train_df, heldout_df = get_datasets(LOCATION+'/data/train.csv')
    train_df, trainOut_df = separateOutput(train_df)
    heldout_df, heldoutOut_df = separateOutput(heldout_df)
    clf = learn(train_df, trainOut_df, heldout_df, heldoutOut_df)

def get_datasets(filepath, train_ratio = 0.80):
    data = pd.read_csv(filepath)
    train_data = data.sample(frac = train_ratio)
    heldout_data = data.loc[~data.index.isin(train_data.index)]
    return train_data, heldout_data

# TODO: Change to Random Forest
def learn(train, target, heldout_data, heldout_target):
    n_est = 200
    depth = 10
    iterations = 1 #number of times testing model
    total = 0
    for i in range(iterations):
        clf = RandomForestClassifier(n_estimators = n_est, max_depth = depth)
        clf.fit(train, target)
        predictData = clf.predict(heldout_data)
        acc_score = accuracy_score(predictData, heldout_target)
        total += acc_score
        print("Accuracy Score: ", acc_score)
    avg_acc = total / iterations
    print("Average accuracy: ", avg_acc)
    logger.log("Number of times ran: " + str(iterations))
    logger.log("Number of trees: " + str(n_est))
    logger.log("Max depth: " + str(depth))
    logger.log("Average accuracy: " + str(avg_acc))
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
