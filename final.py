import argparse
import pandas
import json
import numpy as np
import datetime
from threading import Thread
from DisasterTweets.utility import LOCATION
import DisasterTweets.utility as utility
from DisasterTweets.log import get_logger
from DisasterTweets.models import lr
from DisasterTweets.neural_nets.nn_keras import run as run_keras
#from neural_nets.nn_pytorch import run as run_pytorch
#import utility
ratio = 0.8 # how much of the data is training data. The rest is validation data
path = LOCATION+'/data/'
logger = get_logger('misc')
funcs = {
    'lr': lr.learn,
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


def preprocess_tweets(train_data, result, index):
    logger.log('simplifying training data...')
   # print('thread {} is simplifying data'.format(index))
    simplified_train = [utility.clean_tweet(data[-1]) for data in train_data]
    #print(simplified_train)
    logger.log('tokenizing training data...')
   # print('thread {} is tokenizing data'.format(index))
    tokenized_train = utility.tokenize_tweets(simplified_train)
    result[index] = tokenized_train

num_threads = 10
results = [[]]*num_threads
logger.log('reading in training data...')
train_data = pandas.read_csv(path+'train.csv', header=0)
train_data = train_data.to_numpy() # convert to numpy arrays
logger.log('splitting out labels...')
labels = np.array([data[-1] for data in train_data])
train_data = np.array([data[:4] for data in train_data])
start = datetime.datetime.now()

split = int(len(train_data)/num_threads)+1
print('split:', split)
threads = [Thread(target=preprocess_tweets, args=(train_data[i*split:(i+1)*split], results, i)) for i in range(num_threads)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

end = datetime.datetime.now()
print('preprocessing time with {} threads: {}'.format(num_threads, end-start))

all_tokens = []
for result in results:
    all_tokens += list(result)
all_tokens = np.array(all_tokens)
split = int(len(all_tokens)*ratio)
training_data = all_tokens[:split]
training_labels = labels[:split]
validation_data = all_tokens[split:]
validation_labels = labels[split:]

logger.log('running on the flags', flags)
for flag in flags:
    try:
        funcs[flag](training_data, training_labels, validation_data, validation_labels)
    except KeyError:
        print('Key Error, ignoring flag {}'.format(flag))

if args.filename != None: # if a filename is declared, run the test data too
    logger.log('reading in test data...')
    test_data = pandas.read_csv(path+'test.csv', header=0)
    test_data = test_data.to_numpy()
    logger.log('simplifying test data...')
    simplified_test = [utility.clean_tweet(data[-1]) for data in test_data[:5]]
    logger.log('tokenizing test data...')
    tokenized_test = utility.tokenize_tweets(simplified_test)
