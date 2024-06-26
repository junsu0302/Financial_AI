import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from DNN import create_model, train, test, cols, class_weight

random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

res = []
for optimizer in optimizers:
  model = create_model(optimizer=optimizer)

  t0 = time.time()
  model.fit(train[cols], train['direction'], epochs=50, verbose=False, validation_batch_size=0.2, shuffle=False, class_weight=class_weight)
  t1 = time.time()
  t = t1 - t0

  acc_train = model.evaluate(train[cols], train['direction'], verbose=False)[1]
  acc_test = model.evaluate(test[cols], test['direction'], verbose=False)[1]
  res.append(f'{optimizer:10s} | time(s): {t:.4f} | TRAIN={acc_train:.4f} | TEST={acc_test:.4f}')

for idx in res:
  print(idx)