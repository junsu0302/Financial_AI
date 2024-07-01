import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.regularizers import L1
from keras.src.optimizers import Adam
from sklearn.metrics import accuracy_score

from keras.src.utils.model_visualization import plot_model

# 랜덤 시드 설정
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

# 출력 형식 설정
pd.set_option('display.float_format', '{:.4f}'.format)
np.set_printoptions(suppress=True, precision=4)

# 데이터 로드 및 결측값 제거
symbol = 'EUR='
data = pd.DataFrame(pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()[symbol])

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

  # 시차 변수 생성
  cols = []
  features = [symbol, 'log_return', 'sma', 'rolling_min', 'rolling_max', 'momentum', 'volatility', 'direction']
  for f in features:
    for lag in range(1, lags+1):
      col = f'{f}_lag_{lag}'
      df[col] = df[f].shift(lag)
      cols.append(col)
  df.dropna(inplace=True)
  return df, cols

# 시차 데이터프레임 및 시차 변수 목록
data, cols = add_lags(data, symbol, lags, window=20)

# 모델 생성
optimizer = Adam(learning_rate=0.0001)
def create_model(hidden_layer=2, units=128, dropout=False, rate=0.3, regularize=False, reg=L1(0.0005), optimizer=optimizer, input_dim=len(cols)):
  if not regularize:
    reg = None
  model = Sequential()
  model.add(Dense(hidden_layer, input_dim=input_dim, activity_regularizer=reg, activation='relu'))
  
  if dropout:
    model.add(Dropout(rate, seed=100))
  for _ in range(hidden_layer):
    model.add(Dense(units, activity_regularizer=reg, activation='relu'))
    if dropout:
      model.add(Dropout(rate, seed=100))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model

# 훈련, 테스트 데이터 분할 및 정규화
split = '2018-01-01'
train = data.loc[:split].copy()
test = data.loc[split:].copy()
mu, std = train.mean(), train.std()
train_scaled = (train - mu) / std  # 데이터 정규화
test_scaled = (test - mu) / std    # 데이터 정규화

# 모델 생성 
model = create_model(units=64)
model.fit(train_scaled[cols], train['direction'], epochs=20, verbose=False, validation_split=0.2, shuffle=False)

# 학습 데이터 평가 및 예측, 전략 계산
train_evaluate = model.evaluate(train_scaled[cols], train['direction'])
train['pred'] = np.where(model.predict(train_scaled[cols]) > 0.5, 1, -1)
train['strategy'] = train['pred'] * train['log_return']

# 테스트 데이터 평가 및 예측, 전략 계산
test_evaluate = model.evaluate(test_scaled[cols], test['direction'])
test['pred'] = np.where(model.predict(test_scaled[cols]) > 0.5, 1, -1)
test['strategy'] = test['pred'] * test['log_return']

# 거래 비용 설정
spread = 0.00012
pc = spread / data[symbol].mean()

# 거래 비용을 고려한 전략 수익률 계산
test['strategy_scaled'] = np.where(test['pred'].diff() != 0, test['strategy']-pc, test['strategy'])
test['strategy_scaled'].iloc[0] -= pc
test['strategy_scaled'].iloc[-1] -= pc

# 누적 수익률 그래프 및 훈련, 테스트 데이터 평가 시각화
test[['log_return', 'strategy', 'strategy_scaled']].cumsum().apply(np.exp).plot(figsize=(10, 6))
print('train:', train_evaluate)
print('test:', test_evaluate)
plt.show()