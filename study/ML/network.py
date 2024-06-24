import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from data import f_test, f_train, l_test, l_train, features, labels

def create_model(hidden_layers=2, units=256):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=1))
    for _ in range(hidden_layers-1):
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

model1 = create_model(2, 256)
callbacks = [EarlyStopping(monitor='loss',
                           patience=100,
                           restore_best_weights=True)]

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 8))

ax[0, 0].plot(f_train, l_train, 'ro', label='Training Data')
ax[1, 0].plot(f_test, l_test, 'go', label='Testing Data')
ax[0, 1].plot(f_train, l_train, 'ro', label='Training Data')
ax[1, 1].plot(f_test, l_test, 'go', label='Testing Data')

for i in range(200, 1001, 200):
    model1.fit(f_train, l_train, epochs=200, verbose=False, validation_data=(f_test, l_test), callbacks=callbacks)
    p = model1.predict(f_train)
    ax[0, 0].plot(f_train, p, '--', label=f'Epochs={i}, Hidden Layers=2 (Train)')
    p = model1.predict(f_test)
    ax[1, 0].plot(f_test, p, '--', label=f'Epochs={i}, Hidden Layers=2 (Test)')

for i in range(1, 7, 1):
    model2 = create_model(i, 256)
    model2.fit(f_train, l_train, epochs=200, verbose=False)
    p = model2.predict(f_train)
    ax[0, 1].plot(f_train, p, '--', label=f'Epochs=200, Hidden Layers={i} (Train)')
    p = model2.predict(f_test)
    ax[1, 1].plot(f_test, p, '--', label=f'Epochs=200, Hidden Layers={i} (Test)')

# 제목 및 라벨 추가
ax[0, 0].set_title('Model with 2 Hidden Layers over Epochs')
ax[1, 0].set_title('Model with 2 Hidden Layers over Epochs')
ax[0, 0].set_ylabel('Labels')
ax[1, 0].set_ylabel('Labels')
ax[1, 0].set_xlabel('Features')

ax[0, 1].set_title('Models with Varying Hidden Layers at 200 Epochs')
ax[1, 1].set_title('Models with Varying Hidden Layers at 200 Epochs')
ax[0, 1].set_ylabel('Labels')
ax[1, 1].set_ylabel('Labels')
ax[1, 1].set_xlabel('Features')

ax[0, 0].legend()
ax[1, 0].legend()
ax[0, 1].legend()
ax[1, 1].legend()
plt.tight_layout()
plt.savefig('assets/ML/network.png')
plt.show()
