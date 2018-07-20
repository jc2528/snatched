from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

data1 = pd.read_excel('accel_x_final_dataset.xlsx')

dataset = data1.values

data = dataset[:,0:13].astype(float) # sensor data


#Y = dataset[:,12].astype(int)

print(data1.shape)

#data = np.array(data1)[1:, [0,12]]
#print(data[:,12])
#data = data.tranpose()
print(data.shape)

m = len(data)
train = round(m/2)
valid = train + round(m/4)

test_x = data[valid:m, :-1]
test_y = data[valid:m, -1]

def make_nn(data):

    train_x = data[0:train, :-1]
    train_y = data[0:train, -1]

    val_x = data[train:valid, :-1]
    val_y = data[train:valid, -1]



    model = Sequential()
    model.add(Dense(15, input_dim= 12, activation='relu'))
    #model.add(Dense(15, activation='relu'))
    #model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    #model.add(Dense(1, input_dim = 2, activation = 'relu'))
    model.compile(optimizer= 'SGD', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs = 15, batch_size = 10, validation_data=[val_x, val_y])

    scores = model.evaluate(train_x, train_y)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model

model = make_nn(data)
print(model.outputs)

#test_x = data[valid:m, 0]
#test_y = data[valid:m, -1]
ynew = model.predict_classes(test_x)
#print(sum(test_y))
#print(sum(ynew))








