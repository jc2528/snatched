from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd

data1 = pd.read_excel('sample_data.xlsx')

print(data1.shape)

data = np.array(data1)[2:, [1, 17]]
#print(data.shape)
#data = data.tranpose()
print(data.shape)

m = len(data)
train = round(m/2)
valid = train + round(m/4)

train_x = data[0:train, 0]
train_y = data[0:train, -1]

val_x = data[train:valid, 0]
val_y = data[train:valid, -1]

test_x = data[valid:m, 0]
test_y = data[valid:m, -1]

model = Sequential()
model.add(Dense(2, input_shape = (data.shape[1],1)))
model.compile(optimizer= 'SGD', loss='binary_crossentropy', metrics=['accuracy'])
#model.add(Dense(2))
#print(train_x.shape)
#print(train_y.shape)
model.fit(train_x.reshape(train_x.shape[0], 1), train_y.reshape(train_y.shape[0],1), epochs = 15, batch_size = 10)

scores = model.evaluate(train_x, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





