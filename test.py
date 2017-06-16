from keras.layers import Dense, Activation
from keras.models import Sequential
model = Sequential([
    Dense(32, input_dim=2),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
model.fit([[0,1], [1,1], [1,0]], [1,2,3])
