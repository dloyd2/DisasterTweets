'''
    Matt Briones
    Last Modified: Mar10, 2020
'''


import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from DisasterTweets.log import get_logger
#from DisasterTweets.utility import LOCATION

#logger = get_logger('lr')

testWithRandData = True

#def run():
#    logger.log('running the lr algorithm')

#generate random data for testing, separating out the data and its target output
def gen_rand_data(rows, cols):
    data = pd.DataFrame(np.random.rand(rows, cols-1))
    output = pd.DataFrame(np.rint(np.random.randint(2, size = (rows, 1))), columns = ['out'])
    return pd.concat([data, output], axis = 1)

def get_datasets(filepath, train_ratio = 0.80):
    data = pd.read_csv(filepath)
    train_data = data.sample(frac = train_ratio)
    heldout_data = data.loc[~data.index.isin(train_data.index)]
    return train_data, heldout_data

#train_df, heldout_df = get_datasets(LOCATION+'/data/train.csv')
#print(train_df.head(5))

# TODO: Implement LR algorithm
def learn(train, target, heldout_data, heldout_target):
    clf = SGDClassifier(loss = 'log', max_iter = 1000)
    clf.fit(train, target)
    print(clf.coef_)
    predictData = clf.predict(heldout_data)
    print("Score: ", accuracy_score(predictData, heldout_target))
    return clf

# TODO: Implement
#tests trained model on our heldout data
def test(model, heldout_data, heldout_target):
    return 0

#try using base ML algorithm with randomly generated data
def tryLearnWithRand():
    print("HERE")
    data = gen_rand_data(1000, 5)
    train_df = data.sample(frac = 0.80)
    heldout_df = data.loc[~data.index.isin(train_df.index)]

    trainOut_df = train_df['out']
    heldoutOut_df = heldout_df['out']
    train_df = train_df.drop(['out'], axis = 1)
    print(train_df)
    heldout_df = heldout_df.drop(columns = ['out'])
    learn(train_df, trainOut_df, heldout_df, heldoutOut_df)

if(testWithRandData):
    tryLearnWithRand()
