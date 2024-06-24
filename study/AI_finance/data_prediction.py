import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.src.layers import Dense
from keras.src.models import Sequential

from EMH import add_lags

np.random.seed(100)
tf.random.set_seed(100)

# 데이터 불러오기 및 NaN 값 제거
data = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()

# 로그 수익률 데이터 생성
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)

# 시차 데이터프레임 생성
lags = 7
dfs = {}
for sym in data:
  df, cols = add_lags(rets, sym, lags)
  mu, std = df[cols].mean(), df[cols].std()
  df[cols] = (df[cols] - mu) / std
  dfs[sym] = df

# 회귀 분석을 통한 예측값 계산
OLS_pred = []
for sym in data:
  df = dfs[sym]
  reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0]
  pred = np.dot(df[cols], reg)
  acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
  OLS_pred.append(f'OLS | {sym:10s} | acc={acc:.4f}')

# 신경망 모델 생성
def create_model(problem='regression'):
  model = Sequential()
  model.add(Dense(512, input_dim=len(cols), activation='relu'))
  if problem == 'regression':
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
  else:
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
  return model

# 신경망을 통한 예측값 계산
DNN_pred = []
for sym in data.columns:
  df = dfs[sym]
  model = create_model()
  model.fit(df[cols], df[sym], epochs=25, verbose=False)
  pred = model.predict(df[cols])
  acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
  DNN_pred.append(f'DNN | {sym:10s} | acc={acc:.4f}')

# 결과 시각화
for i in range(len(DNN_pred)):
  print(f'{OLS_pred[i]} || {DNN_pred[i]}')