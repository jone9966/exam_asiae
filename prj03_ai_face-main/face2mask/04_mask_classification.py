import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

X_train, X_test, Y_train, Y_test = np.load('./image/image_data.npy', allow_pickle=True)
print('X_train', X_train.shape)
print('Y_train', Y_train.shape)
print('X_test', X_test.shape)
print('Y_test', Y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(128, 128, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=7)
fit_hist = model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_split=0.15, callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])
model.save('./models/mask_{}.h5'.format(score[1]))
