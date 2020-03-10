import argparse
import pandas
import numpy as np

from DisasterTweets.log import get_logger
from DisasterTweets.models import lr
from DisasterTweets.neural_nets.nn_keras import run as run_keras
#from neural_nets.nn_pytorch import run as run_pytorch
#import utility
logger = get_logger('misc')
funcs = {
    'lr': lr.run,
    'svm': lambda: print('TODO: run svm'),
    'knn': lambda: print('TODO: run knn'),
    'dt': lambda: print('TODO: run dt'),
    'nb': lambda: print('TODO: run nb'),
    'rf': lambda: print('TODO: run rf'),
    'nn': lambda: print('TODO: run nn'),
}

parser = argparse.ArgumentParser()
parser.add_argument('--all', help='run all algorithms', action='store_true')
parser.add_argument('--lr', help='run the logistic regression algorithm', action='store_true')
parser.add_argument('--svm', help='run the support vector machine algorithm', action='store_true')
parser.add_argument('--knn', help='run the k nearest neighbor algorithm', action='store_true')
parser.add_argument('--dt', help='run the decision tree algorithm', action='store_true')
parser.add_argument('--nb', help='run the naive bayes algorithm', action='store_true')
parser.add_argument('--rf', help='run the random forest algorithm', action='store_true')
parser.add_argument('--nn', help='run the neural net algorithm', action='store_true')
parser.add_argument('--filename', help='specify a output filename and runs the test data.')
parser.add_argument('--path', help='path to the data to be used')
args = parser.parse_args()
args_dict = vars(args)

flags = []
if args.all:
    # run all funcs
    logger.log('runing all algorithms')
    # fetch the flags that have a boolean resolution
    # NOTE: this assumes that no other flags are boolean
    flags = [arg for arg in args_dict if (args_dict[arg]==True or args_dict[arg]==False) and arg != 'all']
else:
    # else, only fetch the flags that are True
    flags = [arg for arg in args_dict if args_dict[arg]==True]

logger.log('running on the flags', flags)
for flag in flags:
    try:
        funcs[flag]()
    except KeyError:
        print('Key Error, ignoring flag {}'.format(flag))

import sys
sys.exit()
path = './data/' if args.path == None else args.path
logger.log('reading in data...')
train_data = pandas.read_csv(path+'train.csv', header=0)
test_data = pandas.read_csv(path+'test.csv', header=0)
train_data = train_data.to_numpy() # convert to numpy arrays
test_data = test_data.to_numpy()
logger.log('splitting data...')
labels = np.array([[data[-1]] for data in train_data])
train_data = np.array([data[:4] for data in train_data])
logger.log('simplifying data...')
simplified = [utility.simplify_tweet(data[-1]) for data in train_data]
simplified = [utility.simplify_tweet(data[-1]) for data in test_data]

logger.log('tokenizing data')
tokenized_train = utility.tokenize(simplified)
tokenized_test = utility.tokenize(simplified)
