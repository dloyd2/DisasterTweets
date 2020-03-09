'''
    Daniel Loyd
'''

def run(train, labels, test):
    '''
    data -> numpy.array
    labels -> numpy.array
    '''
    import numpy as np

    #train = np.array(np.array([element for element in data]+[1]) for data in train)
    print('running keras neural net with training data:', train)
    from keras.layers import Dense, Activation
    from keras.models import Sequential
    from keras import optimizers
    import numpy as np

    model = Sequential()
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train, labels)
