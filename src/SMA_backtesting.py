import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.float_format', '{:.4f}'.format)
np.set_printoptions(suppress=True, precision=4)

symbol = 'EUR='
data = pd.DataFrame(pd.read_csv('data/aiif_eikon_eod_data copy.csv', index_col=0, parse_dates=True).dropna()[symbol])

# 이동 평균 계산
data['SMA1'] = data[symbol].rolling(42).mean()
data['SMA2'] = data[symbol].rolling(258).mean()
data.dropna(inplace=True)

# 단순 이동평균 전력
data['trading_signal'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
data['trading_signal'] = data['trading_signal'].shift(1)
data.dropna(inplace=True)

# 로그 수익률 계산
data['log_return'] = np.log(data[symbol] / data[symbol].shift(1))
data.dropna(inplace=True)

# 수수료를 고려한 전략 수익률 계산
pc = 0.005
data['strategy_return'] = data['trading_signal'] * data['log_return']
data['strategy_return'] = np.where(data['trading_signal'].diff() != 0, data['strategy_return'] - pc, data['strategy_return'])
data.loc[data.index[0], 'strategy_return'] -= pc
data.loc[data.index[-1], 'strategy_return'] -= pc

# 서브플롯 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 첫 번째 그래프: 가격 데이터 및 트레이딩 신호
data[[symbol, 'SMA1', 'SMA2']].plot(ax=ax1)
ax1.set_title('Price and Moving Averages')
ax1.set_ylabel('Price')
ax1_right = ax1.twinx()
data['trading_signal'].plot(ax=ax1_right, style='g--', alpha=0.3, label='Trading Signal')
ax1_right.set_ylabel('Trading Signal')
ax1_right.legend(loc='upper left')

# 두 번째 그래프: 전략 수익률과 로그 수익률의 누적 곡선
data[['log_return', 'strategy_return']].cumsum().apply(np.exp).plot(ax=ax2)
ax2.set_title('Cumulative Returns')
ax2.set_ylabel('Cumulative Return')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.savefig('assets/SMA_backtesting')
plt.show()