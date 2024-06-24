import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기 및 NaN 값 제거
data = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()

# lags 설정
lags = 7
def add_lags(data, ric, lags):
  """
  주어진 데이터에 대해 시차(lag)를 추가합니다.
    
  Args:
    data (pd.DataFrame): 원본 데이터.
    ric (str): 컬럼 이름.
    lags (int): 시차의 수.
    
  Returns:
    df (pd.DataFrame): 시차가 추가된 데이터프레임.
    cols (list): 추가된 시차 컬럼 이름 목록.
  """
  cols = []
  df = pd.DataFrame(data[ric])
  for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    df[col] = df[ric].shift(lag)
    cols.append(col)
  df.dropna(inplace=True)
  return df, cols

# 시차 데이터프레임 생성
dfs = {}
for sym in data.columns:
  df, cols = add_lags(data, sym, lags)
  dfs[sym] = df

# 회귀 분석 수행
regs = {}
for sym in data.columns:
  df = dfs[sym]
  reg = np.linalg.lstsq(df[cols], df[sym], rcond=None)[0]
  regs[sym] = reg

# 회귀 계수 데이터프레임 생성
rega = np.stack(tuple(regs.values()))
regd = pd.DataFrame(rega, columns=cols, index=data.columns)

def plot():
  # 그래프 생성
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))

  # 데이터 정규화 및 시각화
  normalized_data = data / data.iloc[0]
  normalized_data.plot(ax=axes[0])
  axes[0].set_title('Normalized Data')
  axes[0].set_xlabel('Date')
  axes[0].set_ylabel('Normalized Value')
  axes[0].legend(loc='best')

  # 회귀 계수 평균 시각화
  regd.mean().plot(kind='bar', ax=axes[1])
  axes[1].set_title('Mean Regression Coefficients')
  axes[1].set_xlabel('Lags')
  axes[1].set_ylabel('Coefficient Value')

  plt.tight_layout()
  plt.savefig('assets/AI_finance/EMH.png')
  plt.show()
