'''
    Daniel Loyd
'''

def run(train, labels, test):
    '''
    data -> numpy.array
    labels -> numpy.array
    '''
    import numpy as np
    length = max([len(data) for data in train])
    data = []
    for info in train:
        new_info = info
        diff = length - len(new_info)
        if diff > 0:
            buffer = np.zeros(diff, np.int8)
            #print('about to concatenate {} and {}'.format(info, buffer))
            new_info = np.append(info, buffer)
            #print('result: {}'.format(new_info))
        data.append(new_info)
    data = np.array(data)        

    from keras.layers import Dense, Activation
    from keras.models import Sequential
    from keras import optimizers
    import numpy as np
    print('running keras neural net')

    model = Sequential()
    model.add(Dense(1, input_shape=(26,)))
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # data = np.random.random((1000, 100))
    # labels = np.random.randint(2, size=(1000, 1))
   # print('data:', data)
    #print('labels:', type(labels))
    model.fit(data, labels, epochs=1)
