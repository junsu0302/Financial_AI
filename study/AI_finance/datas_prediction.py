import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.src.layers import Dense
from keras.src.models import Sequential
from sklearn.ensemble import BaggingClassifier

np.random.seed(100)
tf.random.set_seed(100)

# 데이터 불러오기 및 NaN 값 제거
data = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()

def add_lags(data, ric, lags, window=50):
  df = pd.DataFrame(data[ric])
  df.dropna()
  df['log_return'] = np.log(df / df.shift(1))               # 로그 수익률 (Log Return)
  df['sma'] = df[ric].rolling(window).mean()                # 이동평균 (Simple Moving Average)
  df['rolling_min'] = df[ric].rolling(window).min()         # 이동최소 (Rolling Minimum)
  df['rolling_max'] = df[ric].rolling(window).max()         # 이동최대 (Rolling Maximum)
  df['momentum'] = df['log_return'].rolling(window).mean()  # 운동량 (Mometum)
  df['volatility'] = df['log_return'].rolling(window).std() # 이동분산 (Volatility)
  df['direction'] = np.where(df['log_return'] > 0, 1, 0)    # 방향 (Direction)
  df.dropna(inplace=True)

  cols = []
  features = [ric, 'log_return', 'sma', 'rolling_min', 'rolling_max', 'momentum', 'volatility', 'direction']
  for f in features:
    for lag in range(1, lags+1):
      col = f'{f}_lag_{lag}'
      df[col] = df[f].shift(lag)
      cols.append(col)
  df.dropna(inplace=True)
  return df, cols

lags = 5
dfs = {}
for ric in data:
  df, cols = add_lags(data, ric, lags)
  dfs[ric] = df.dropna(), cols

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

# 전체 데이터를 통한 평가
IN_SAMPLE = []
for ric in data:
  model = create_model('classification')
  df, cols = dfs[ric]
  df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()        # 표준화
  model.fit(df[cols], df['direction'], epochs=50, verbose=False)  # 모델 훈련
  pred = np.where(model.predict(df[cols]) > 0.5, 1, 0)            # 예측값 이진화
  acc = accuracy_score(df['direction'], pred)                     # 정확도 계산
  IN_SAMPLE.append(f'IN-SAMPLE | {ric:7s} | acc={acc:4f}')

# 훈련-테스트 데이터를 통한 평가
OUT_OF_SAMPLE = []
def train_test_model(model):
  for ric in data:
    # 데이터 분할
    df, cols = dfs[ric]
    split = int(len(df) * 0.85)
    
    # 훈련 데이터로 학습
    train = df.iloc[:split].copy()
    mu, std = train[cols].mean(), train[cols].std()
    train[cols] = (train[cols] - mu) / std
    model.fit(train[cols], train['direction'])
    
    # 데스트 데이터로 평가
    test = df.iloc[split:].copy()
    test[cols] = (test[cols] - mu) / std
    pred = np.where(model.predict(test[cols]) > 0.5, 1, 0)
    acc = accuracy_score(test['direction'], pred)
    OUT_OF_SAMPLE.append(f'OUT-OF-SAMPLE | {ric:7s} | acc={acc:4f}')

model = create_model('classification')
train_test_model(model)

for i in range(len(IN_SAMPLE)):
  print(f'{IN_SAMPLE[i]} || {OUT_OF_SAMPLE[i]}')