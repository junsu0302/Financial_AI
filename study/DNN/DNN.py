import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.layers import Dense, Dropout
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from keras.src.regularizers import L2

from sklearn.metrics import accuracy_score

random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

# 데이터 준비
raw = pd.read_csv('data/aiif_eikon_id_eur_usd.csv', index_col=0, parse_dates=True)
symbol = 'EUR_USD'

data = pd.DataFrame(raw['CLOSE'].loc[:])
data.columns = [symbol]
data = data.resample('1h', label='right').last().ffill()

# 시차 데이터프레임 생성
lags = 5
def add_lags(data, symbol, lags, window=50):
  df = data.copy()
  df.dropna(inplace=True)
  df['log_return'] = np.log(df / df.shift(1))               # 로그 수익률 (Log Return)
  df['sma'] = df[symbol].rolling(window).mean()             # 이동평균 (Simple Moving Average)
  df['rolling_min'] = df[symbol].rolling(window).min()      # 이동최소 (Rolling Minimum)
  df['rolling_max'] = df[symbol].rolling(window).max()      # 이동최대 (Rolling Maximum)
  df['momentum'] = df['log_return'].rolling(window).mean()  # 운동량 (Momentum)
  df['volatility'] = df['log_return'].rolling(window).std() # 이동분산 (Volatility)
  df['direction'] = np.where(df['log_return'] > 0, 1, 0)    # 방향 (Direction)
  df.dropna(inplace=True)

  cols = []
  features = [symbol, 'log_return', 'sma', 'rolling_min', 'rolling_max', 'momentum', 'volatility', 'direction']
  for f in features:
    for lag in range(1, lags+1):
      col = f'{f}_lag_{lag}'
      df[col] = df[f].shift(lag)
      cols.append(col)
  df.dropna(inplace=True)
  return df, cols

data, cols = add_lags(data, symbol, lags)

# 가중치 계산
def get_weight(df):
  c0, c1 = np.bincount(df['direction'])
  w0 = (1 / c0) * len(df) / 2
  w1 = (1 / c1) * len(df) / 2
  return {0: w0, 1: w1}

class_weight = get_weight(data)

# 모델 생성
optimizer = Adam(learning_rate=0.001)
def create_model(hidden_layer=2, units=128, dropout=True, rate=0.3, regularize=True, reg=L2(0.001), optimizer=optimizer, input_dim=len(cols)):
  if not regularize:
    reg = None
  model = Sequential()
  model.add(Dense(units, input_dim=input_dim, activity_regularizer=reg, activation='relu'))

  if dropout:
    model.add(Dropout(rate, seed=100))
  for _ in range(hidden_layer):
    model.add(Dense(units, activation='relu', activity_regularizer=reg))
    if dropout:
      model.add(Dropout(rate, seed=100))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model

# 데이터 분할
split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()

# 가우스 정규화 진행
mu, std = train[cols].mean(), train[cols].std()
train[cols] = (train[cols] - mu) / std
test[cols] = (test[cols] - mu) / std

# 모델 학습
model = create_model()
history = model.fit(train[cols], train['direction'], epochs=50, verbose=1, validation_split=0.2, shuffle=False, class_weight=class_weight)
test['predicted_direction'] = np.where(model.predict(test[cols]) > 0.5, 1, 0)

train_eval = model.evaluate(train[cols], train['direction'])
test_eval = model.evaluate(test[cols], test['direction'])

res = pd.DataFrame(history.history)

# # 시각화
# print(f'Train Evaluate - Loss: {train_eval[0]:.4f}, Accuracy: {train_eval[1]:.4f}')
# print(f' Test Evaluate - Loss: {test_eval[0]:.4f}, Accuracy: {test_eval[1]:.4f}')

# fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# # 학습 및 검증 정확도 시각화
# axs[0].plot(res['accuracy'], label='Train Accuracy')
# axs[0].plot(res['val_accuracy'], label='Validation Accuracy')
# axs[0].set_title('Model Accuracy')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Accuracy')
# axs[0].legend()

# # 학습 및 검증 손실 시각화
# axs[1].plot(res['loss'], label='Train Loss')
# axs[1].plot(res['val_loss'], label='Validation Loss')
# axs[1].set_title('Model Loss')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('Loss')
# axs[1].legend()

# plt.tight_layout()
# plt.show()