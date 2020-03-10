import argparse
import pandas
import numpy as np

from neural_nets.nn_keras import run as run_keras
#from neural_nets.nn_pytorch import run as run_pytorch
import utility
parser = argparse.ArgumentParser()
parser.add_argument('--all', help='run all algorithms')
parser.add_argument('--lr', help='run the logistic regression algorithm')
parser.add_argument('--svm', help='run the support vector machine algorithm')
parser.add_argument('--knn', help='run the k nearest neighbor algorithm')
parser.add_argument('--dt', help='run the decision tree algorithm')
parser.add_argument('--nb', help='run the naive bayes algorithm')
parser.add_argument('--rf', help='run the random forest algorithm')
parser.add_argument('--nn', help='run the neural net algorithm')
parser.add_argument('--test', help='produce a prediction for the test data and store to a file')

parser.add_argument('--filename', help='if test flag is enabled, specify the name of the output file.')
parser.add_argument('--path', help='path to the data to be used')
args = parser.parse_args()

if args.t == None:
    raise Exception('must be run with the -t flag. For more help run: python3 final.py -h ')

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