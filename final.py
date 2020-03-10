import argparse
import pandas
import numpy as np

from neural_nets.nn_keras import run as run_keras
#from neural_nets.nn_pytorch import run as run_pytorch
#import utility

funcs = {
    'lr': lambda: print('run lr'),
    'svm': lambda: print('run svm'),
    'knn': lambda: print('run knn'),
    'dt': lambda: print('run dt'),
    'nb': lambda: print('run nb'),
    'rf': lambda: print('run rf'),
    'nn': lambda: print('run nn'),
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
#parser.add_argument('--test', help='produce a prediction for the test data and store to a file', action='store_true')
parser.add_argument('--filename', help='specify a output filename and runs the test data.')
parser.add_argument('--path', help='path to the data to be used')
args = parser.parse_args()
args_dict = vars(args)

flags = []
if args.all:
    # run all funcs
    # raise Exception('must be run with the -t flag. For more help run: python3 final.py -h ')
    print('run all')
    flags = [arg for arg in args_dict if (args_dict[arg]==True or args_dict[arg]==False) and arg != 'all']

else:
    flags = [arg for arg in args_dict if args_dict[arg]==True]
print(flags)
for flag in flags:
    funcs[flag]()
import sys
sys.exit()
path = './data/' if args.path == None else args.path
print('reading in data...')
train_data = pandas.read_csv(path+'train.csv', header=0)
test_data = pandas.read_csv(path+'test.csv', header=0)
train_data = train_data.to_numpy() # convert to numpy arrays
test_data = test_data.to_numpy()
print('splitting data...')
labels = np.array([[data[-1]] for data in train_data])
train_data = np.array([data[:4] for data in train_data])
print('simplifying data...')
simplified = [utility.simplify_tweet(data[-1]) for data in train_data]
simplified = [utility.simplify_tweet(data[-1]) for data in test_data]

print('tokenizing data')
tokenized_train = utility.tokenize(simplified)
tokenized_test = utility.tokenize(simplified)

exec("func = run_{}".format(args.t.lower()))
func(tokenized_train, labels, tokenized_test)