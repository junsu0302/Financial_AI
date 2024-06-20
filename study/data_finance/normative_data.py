import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 데이터 로드 및 심볼 설정
url = 'data/aiif_eikon_eod_data.csv'
raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
symbols = ['AAPL.O', 'MSFT.O', 'INTC.O', 'AMZN.O', 'GLD']

#! 평균-분산 포트폴리오 이론

def port_return(rets, weights):
  """포트폴리오의 연간 기대 수익률 계산"""
  return np.dot(rets.mean(), weights) * 252

def port_volatility(rets, weights):
  """포트폴리오의 연간 변동성 계산"""
  return np.dot(weights, np.dot(rets.cov() * 252, weights)) ** 0.5

def port_sharpe(rets, weights):
  """포트폴리오의 샤프 비율 계산"""
  return port_return(rets, weights) / port_volatility(rets, weights)

rets = np.log(raw[symbols] / raw[symbols].shift(1)).dropna() # 로그 수익률
weights = len(rets.columns) * [1 / len(rets.columns)] # 초기 포트폴리오 가중치 설정

# 초기 포트폴리오 성과 출력
print(f"Initial Portfolio Return: {port_return(rets, weights):.2f}")
print(f"Initial Portfolio Volatility: {port_volatility(rets, weights):.2f}")
print(f"Initial Portfolio Sharpe Ratio: {port_sharpe(rets, weights):.2f}")

# 임의 포트폴리오 시뮬레이션
w = np.random.random((1000, len(symbols)))
w = (w.T / w.sum(axis=1)).T

# 임의 포트폴리오 변동성 및 수익률 계산
pvr = [(port_volatility(rets[symbols], weights), port_return(rets[symbols], weights)) for weights in w]
pvr = np.array(pvr)

# 임의 포트폴리오 샤프 비율 계산
psr = pvr[:, 1] / pvr[:, 0]

# 이미지 생성
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# 임의 포트폴리오 성과 시각화
scatter = axes[0, 0].scatter(pvr[:, 0], pvr[:, 1], c=psr, cmap='coolwarm')
cb = fig.colorbar(scatter, ax=axes[0, 0])
cb.set_label('Sharpe ratio')
axes[0, 0].set_xlabel('Expected Volatility')
axes[0, 0].set_ylabel('Expected Return')
axes[0, 0].set_title('Random Portfolio Performance')

# 포트폴리오 최적화 설정
bnds = len(symbols) * [(0, 1)]
cons = {'type': 'eq', 'fun': lambda weights: weights.sum() - 1}

# 연도별 최적 포트폴리오 가중치 계산 (샤프 비율이 최대)
opt_weights = {}
for year in range(2010, 2019):
  rets_ = rets[symbols].loc[f'{year}-01-01': f'{year}-12-31']
  result = minimize(lambda weights: -port_sharpe(rets_, weights),
                    weights,
                    bounds=bnds,
                    constraints=cons)
  opt_weights[year] = result['x']

# 연도별 예상 및 실제 성과 계산
res_list = []
for year in range(2010, 2019):
  rets_ = rets[symbols].loc[f'{year}-01-01': f'{year}-12-31']
  next_rets_ = rets[symbols].loc[f'{year+1}-01-01': f'{year+1}-12-31']
    
  # 기대 포트폴리오
  epv = port_volatility(rets_, opt_weights[year])
  epr = port_return(rets_, opt_weights[year])
  esr = epr / epv
    
  # 실제 포트폴리오
  rpv = port_volatility(next_rets_, opt_weights[year])
  rpr = port_return(next_rets_, opt_weights[year])
  rsr = rpr / rpv
    
  res_list.append(pd.DataFrame({'epv': epv, 'epr': epr, 'esr': esr,
                                'rpv': rpv, 'rpr': rpr, 'rsr': rsr}, index=[year+1]))

res = pd.concat(res_list)

# 기대변동성과 실제변동성 비교 시각화
res[['epv', 'rpv']].plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Expected vs Realized Portfolio Volatility (Yearly)')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Volatility')

# 기대 샤프비율과 실제 샤프비율 비교 시각화
res[['esr', 'rsr']].plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Expected vs Realized Sharpe Ratio (Yearly)')
axes[0, 2].set_xlabel('Year')
axes[0, 2].set_ylabel('Sharpe Ratio')

#! 자본자산 가격결정 모형 (CAPM)

# 데이터 준비
r = 0.005
market = '.SPX'
rets = np.log(raw / raw.shift(1)).dropna()

res_list = []
for sym in rets.columns[:4]:
  for year in range(2010, 2019):
    # 해당 연도 수익률 데이터
    rets_ = rets.loc[f'{year}-01-01': f'{year}-12-31']
    muM = rets_[market].mean() * 252    # 연간 시장 평균 수익률
    cov = rets_.cov().loc[sym, market]  # 주식과 시장 간의 공분산
    var = rets_[market].var()           # 시장 수익률의 분산
    beta = cov / var                    # 베타
        
    # 다음 연도 수익률 데이터
    next_rets_ = rets.loc[f'{year+1}-01-01': f'{year+1}-12-31']
    muM = next_rets_[market].mean() * 252 # 다음 연도 시장 평균 수익률

    # 수익률 계산
    mu_capm = r + beta * (muM - r)          # CAPM을 사용한 예상 수익률
    mu_real = next_rets_[sym].mean() * 252  # 실제 수익률
        
    res_list.append(pd.DataFrame({'symbol': sym,
                                  'mu_capm': mu_capm,
                                  'mu_real': mu_real}, index=[year+1]))

res = pd.concat(res_list)

# 특정 주식의 연도별 CAPM 결과 시각화
target_sym = 'AAPL.O'
res[res['symbol'] == target_sym].plot(kind='bar', ax=axes[0, 3])
axes[0, 3].set_title(f'{target_sym} CAPM Expected vs Realized Returns (Yearly)')
axes[0, 3].set_xlabel('Year')
axes[0, 3].set_ylabel('Return')

# 전체 주식의 평균 CAPM 결과 시각화
grouped = res.groupby('symbol').mean()
grouped.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Average CAPM Expected vs Realized Returns (2010-2019)')
axes[1, 0].set_xlabel('Symbol')
axes[1, 0].set_ylabel('Return')

#! 차익거래 가격결정 이론 (APT)

factors = ['.SPX', '.VIX', 'EUR=', 'XAU='] # 요인

res_list = []
np.set_printoptions(formatter={'float': lambda x: f'{x:5.2f}'})
for sym in rets.columns[:4]:
  for year in range(2010, 2019):
    # 현재 연도 수익률 데이터
    rets_ = rets.loc[f'{year}-01-01': f'{year}-12-31']
    reg = np.linalg.lstsq(rets_[factors], rets_[sym], rcond=1)[0]
        
    # 다음 연도 수익률 데이터
    next_rets_ = rets.loc[f'{year+1}-01-01': f'{year+1}-12-31']
    mu_apt = np.dot(next_rets_[factors].mean() * 252, reg)
    mu_real = next_rets_[sym].mean() * 252
    
    res_list.append(pd.DataFrame({'symbol': sym,
                                  'mu_apt': mu_apt,
                                  'mu_real': mu_real}, index=[year+1]))

res = pd.concat(res_list)

# 특정 주식의 연도별 APT 결과 시각화
res[res['symbol'] == target_sym].plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title(f'{target_sym} APT Expected vs Realized Returns (Yearly)')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Return')

# 전체 주식의 평균 APT 결과 시각화
grouped = res.groupby('symbol').mean()
grouped.plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Average APT Expected vs Realized Returns (2010-2019)')
axes[1, 2].set_xlabel('Symbol')
axes[1, 2].set_ylabel('Return')

# 추가 분석
factors = pd.read_csv('data/aiif_eikon_eod_factors.csv', index_col=0, parse_dates=True)

start = '2017-01-01'
end = '2020-01-01'

retsd = rets.loc[start:end].copy()
retsd.dropna(inplace=True)

retsf = np.log(factors / factors.shift(1))
retsf = retsf.loc[start:end]
retsf.dropna(inplace=True)
retsf = retsf.loc[retsd.index].dropna()
retsf_corr = retsf.corr()

# 요인 수익률 상관관계 시각화
cax = axes[1, 3].matshow(retsf_corr)
fig.colorbar(cax, ax=axes[1, 3])
axes[1, 3].set_xticks(range(len(retsf_corr.columns)))
axes[1, 3].set_xticklabels(retsf_corr.columns, rotation=90)
axes[1, 3].set_yticks(range(len(retsf_corr.columns)))
axes[1, 3].set_yticklabels(retsf_corr.columns)
axes[1, 3].set_title('Correlation of Factor Returns (2017-2020)')

plt.tight_layout()
plt.savefig('assets/normative_data.png')
plt.show()
