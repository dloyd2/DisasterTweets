import pandas as pd
import numpy as np

# separate the target and the data into different datasets
def separateOutput(data):
    dataOut = data['out']
    data = data.drop(columns = ['out'])
    return data, dataOut

# generate random data for testing, separating out the data and its target output
def gen_rand_data(rows, cols):
    data = pd.DataFrame(np.random.rand(rows, cols-1))
    output = pd.DataFrame(np.rint(np.random.randint(2, size = (rows, 1))), columns = ['out'])
    return pd.concat([data, output], axis = 1)
