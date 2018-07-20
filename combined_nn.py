from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Concatenate, Input
import numpy as np
import pandas as pd

data1 = pd.read_excel('sample_data.xlsx')
data1 = np.array(data1)[2:, 1:]

print(data1.shape)

data = []

for i in range(6):
    #data.append(np.array(data1[1:, [i,12]]))
    data.append(np.array(data1[:,i]))
#print(data.shape)

m = len(data[1])
train = round(m/2)
valid = train + round(m/4)

models = []

train_x = []
val_x = []
test_x = []

train_y = data1[0:train, -1]
val_y = data1[train:valid, -1]
test_y = data1[valid:m, -1]

print(test_y)

for i in range(6):
    #print(data[i].shape)
    train_x.append((data[i])[0:train])
    val_x.append((data[i])[train:valid])
    test_x.append((data[i])[valid:m])

    in2 = Input(shape=(1,))
    #print(in2)
    model_two_dense_1 = Dense(10, activation='relu')(in2)
    models.append(in2)
    #model_two_dense_2 = Dense(128, activation='relu')(model_two_dense_1)


    # model = Sequential()
    # model.add(Dense(1, input_dim= 1, activation= 'relu'))
    # models.append(model)
    #model.add(Dense(2))


    #print(train_x.shape)
    #print(train_y.shape)
    #

model_final_concat = Concatenate(axis = -1)(models)
model_final = Dense(1, activation = 'softmax')(model_final_concat)
model = Model(inputs = models, outputs = model_final)
model.compile(optimizer= 'SGD', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs = 10, batch_size = 10, validation_data=(val_x, val_y))

scores = model.evaluate(train_x, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

ynew = model.predict(test_x)