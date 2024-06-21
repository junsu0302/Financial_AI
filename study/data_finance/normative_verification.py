import math
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# 표준정규분포 데이터 생성
N = 10000
snrn = np.random.standard_normal(N)
snrn -= snrn.mean()
snrn /= snrn.std()

# 1,2차 모멘트를 갖는 데이터 생성
numbers = np.ones(N) * 1.5
split = int(0.25 * N)
numbers[split:3*split] = -1
numbers[3*split:4*split] = 0
numbers -= numbers.mean()
numbers /= numbers.std()

def PDF(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
  """
  정규분포의 확률 밀도 함수(PDF)를 계산한다.

  정규분포는 특정 평균(mu)과 표준편차(sigma)를 가지는 연속 확률 분포이다.
  확률 밀도 함수(PDF)는 주어진 값 x가 정규분포 내에서 발생할 확률의 밀도를 나타낸다.

  Args:
    x (np.ndarray): PDF를 평가할 값
    mu (np.ndarray): 정규분포의 평균
    sigma (np.ndarray): 정규분포의 표준편차

  Returns:
    np.ndarray: 입력 값 x에 대한 PDF 값
  """
  z = (x - mu) / sigma
  pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
  return pdf

def return_histogram(ax:plt.Axes, rets: np.ndarray, title: str = ''):
  """
  동일한 평균과 표준편차를 갖는 수익률의 히스토그램과 정규분포의 확률밀도함수를 그림

  수익률 데이터가 해당 정규분퍼를 얼마나 잘 따르는지 확인할 수 있다.
  히스토그램은 주어진 데이터의 빈도를 나타내고, PDF는 이론적인 정규분포를 나타낸다.

  Args:
    ax (plt.Axes): matplotlib Axes 객체
    rets (np.ndarray): 로그 수익률 데이터
    title (str, optional): 플롯의 제목 (기본값은 '')
  """
  x = np.linspace(min(rets), max(rets), 100)
  y = PDF(x, np.mean(rets), np.std(rets))
  ax.hist(np.array(rets), bins=50, density=True, label='frequency')
  ax.plot(x, y, linewidth=2, label='PDF')
  ax.set_xlabel('log returns')
  ax.set_ylabel('frequency/probability')
  ax.set_title(title)
  ax.legend()

def return_qqplot(ax:plt.Axes, rets: np.ndarray, title: str = ''):
  """
  수익률 데이터의 QQ 플롯을 생성한다.

  QQ 플롯은 두 확률분포를 비교하는 그래프다.
  주어진 데이터가 정규분포를 얼마나 잘 따르는지를 시각적으로 확인할 수 있게 해준다.
  이론적 분위수와 표본 분위수를 비교하여, 데이터가 정규분포와 얼마나 유사한지 알 수 있다.

  Args:
    ax (plt.Axes): matplotlib Axes 객체
    rets (np.ndarray): 로그 수익률 데이터
    title (str, optional): 플롯의 제목 (기본값은 '')
  """
  sm.qqplot(rets, line='s', alpha=0.5, ax=ax)
  ax.set_title(title)
  ax.set_xlabel('theoretical quantiles')
  ax.set_ylabel('sample quantiles')

def print_statistics(rets: np.ndarray):
  """수익률 데이터의 다양한 통계적 지표를 출력한다.

  이 함수는 주어진 수익률 데이터에 대한 통계적 특성을 출력한다.
  왜도(Skewness), 첨도(Kurtosis), 정규성 테스트 결과를 포함한다.
  - 왜도: 데이터의 비대칭성을 나타낸다. 왜도가 0이면 완전한 대칭을 의미한다.
  - 첨도: 데이터의 뾰족함을 나타낸다. 첨도가 0이면 정규분포와 유사한 뾰족함을 의미한다.
  - 정규성 테스트: 데이터가 정규분포를 따르는지 여부를 판단하는 테스트이다.

  Args:
    rets (np.ndarray): 로그 수익률 데이터
  """
  print('\033[35m')
  print('        RETURN SAMPLE STATISTICS        ')
  print('-' * 40)
  print('%30s %9.6f' % ('표본 로그 수익률의 왜도 | ', scs.skew(rets)))
  print('%30s %9.6f' % ('왜도 정규성 검정 p-값 | ', scs.skewtest(rets)[1]))
  print('-' * 40)
  print('%30s %9.6f' % ('표본 로그 수익률의 첨도 | ', scs.kurtosis(rets)))
  print('%30s %9.6f' % ('첨도 정규성 검정 p-값 | ', scs.kurtosistest(rets)[1]))
  print('-' * 40)
  print('%30s %9.6f' % ('정규성 검정 p-값 | ', scs.normaltest(rets)[1]))
  print('-' * 40)
  print('\033[0m')

# 현실의 수익률 데이터
raw = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
rets = np.log(raw / raw.shift(1)).dropna()

r = 0.005       # 무위험 이자율
market = '.SPX' # 시장 지수

res_list = []
for sym in rets.columns[:4]:
  for year in range(2010, 2019):
    # 해당 연도 수익률 데이터
    rets_ = rets.loc[f'{year}-01-01': f'{year}-12-31']
    muM = rets_[market].mean() * 252
    var = rets_[market].var()
    cov = rets_.cov().loc[sym, market]
    beta = cov / var

    # 다음 연도 수익률 데이터
    next_rets_ = rets.loc[f'{year+1}-01-01': f'{year+1}-12-31']
    muM = next_rets_[market].mean() * 252
    
    mu_capm = r + beta * (muM - r)          # CAPM 예상 수익률
    mu_real = next_rets_[sym].mean() * 252  # 실제 수익률
    res_list.append(pd.DataFrame({'symbol': sym,
                                  'beta': beta,
                                  'mu_capm': mu_capm,
                                  'mu_real': mu_real},
                                  index=[year + 1]))
res = pd.concat(res_list)

# 베타 값과 CAPM 예상 수익률 간의 회귀 분석
reg_capm = np.polyfit(res['beta'], res['mu_capm'], deg=1)
res['mu_capm_ols'] = np.polyval(reg_capm, res['beta'])

# 베타 값과 실제 수익률 간의 회귀 분석
reg_real = np.polyfit(res['beta'], res['mu_real'], deg=1)
res['mu_real_ols'] = np.polyval(reg_real, res['beta'])

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

return_histogram(axes[0, 0], rets['AAPL.O'].values, 'AAPL.O')
return_qqplot(axes[0, 1], rets['AAPL.O'].values, 'AAPL.O')

res.plot(kind='scatter', x='beta', y='mu_capm', ax=axes[1, 0])
x = np.linspace(res['beta'].min(), res['beta'].max())
axes[1, 0].plot(x, np.polyval(reg_capm, x), 'g--', label='regression')
axes[1, 0].legend()
axes[1, 0].set_title('CAPM Expected Return vs Beta')

res.plot(kind='scatter', x='beta', y='mu_real', ax=axes[1, 1])
axes[1, 1].plot(x, np.polyval(reg_real, x), 'g--', label='regression')
axes[1, 1].legend()
axes[1, 1].set_title('Realized Return vs Beta')

plt.tight_layout()
plt.savefig('assets/data_finance/normative_verification.png')
plt.show()
