import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

#? 금융 상품 가격 설정 (0: 현재, 1: 만기)
S0, S1 = 10, np.array((20, 10, 5))
T0, T1 = 10, np.array((1, 12, 13))

#? 자산 벡터 설정 (0: 현재, 1: 만기)
M0 = np.array((S0, T0))
M1 = np.array((S1, T1)).T

#? 시나리오 확률 설정
P = np.ones(3) / 3

#? 초기 포트폴리오 배분 설정
phi = np.array((0.5, 0.5))

# TODO: 포트폴리오 통계
#? 전체 자산의 수익률
rM = M1 / M0 - 1

#? 자산 기대 수익률
def mu(rX):
  return np.dot(P, rX)

#? 자산 분산
def var(rX):
  return ((rX - mu(rX)) ** 2).mean()

#? 자산 변동성
def sigma(rX):
  return np.sqrt(var(rX))

#? 포트폴리오 기대 수익률
def mu_phi(phi):
  return np.dot(phi, mu(rM))

#? 포트폴리오 분산
def var_phi(phi):
  cv = np.cov(rM.T, aweights=P, ddof=0)
  return np.dot(phi, np.dot(cv, phi))

#? 포트폴리오 변동성
def sigma_phi(phi):
  return var_phi(phi) ** 0.5

# TODO: 투자 기회 집합  
phi_mcs = np.random.random((2, 200))
phi_mcs = (phi_mcs / phi_mcs.sum(axis=0)).T
mcs = np.array([(sigma_phi(phi), mu_phi(phi)) for phi in phi_mcs])

# plt.figure(figsize=(10, 6))
# plt.plot(mcs[:, 0], mcs[:, 1], 'ro')
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.show()

# TODO: 최소 변동성 및 최대 샤프 비율
cons = {
  'type': 'eq', 
  'fun': lambda phi: np.sum(phi) - 1
}
bnds = ((0, 1), (0, 1))

min_var = minimize(sigma_phi, (0.5, 0.5), constraints=cons, bounds=bnds)

def sharpe(phi):
  return mu_phi(phi) / sigma_phi(phi)

max_sharpe = minimize(lambda phi: -sharpe(phi), (0.5, 0.5), constraints=cons, bounds=bnds)

print(min_var)
print(max_sharpe)

# plt.figure(figsize=(10, 6))
# plt.plot(mcs[:, 0], mcs[:, 1], 'ro', ms=5)
# plt.plot(sigma_phi(min_var['x']), mu_phi(min_var['x']), '^', ms=12.5, label='minimum volatility')
# plt.plot(sigma_phi(max_sharpe['x']), mu_phi(max_sharpe['x']), 'v', ms=12.5, label='maximum Sharpe ratio')
# plt.xlabel('expected volalitity')
# plt.ylabel('expected return')
# plt.legend()
# plt.show()

# TODO: 효율적 투자 경계선

cons = [
  {'type': 'eq', 'fun': lambda phi: np.sum(phi) - 1},
  {'type': 'eq', 'fun': lambda phi: mu_phi(phi) - target}
]
bnds = ((0, 1), (0, 1))

targets = np.linspace(mu_phi(min_var['x']), 0.16)
frontier = []
for target in targets:
  phi_eff = minimize(sigma_phi, (0.5, 0.5), constraints=cons, bounds=bnds)['x']
  frontier.append((sigma_phi(phi_eff), mu_phi(phi_eff)))

frontier = np.array(frontier)

plt.figure(figsize=(10, 6))
plt.plot(frontier[:, 0], frontier[:, 1], 'mo', ms=5, label='efficient frontier')
plt.plot(sigma_phi(min_var['x']), mu_phi(min_var['x']), '^', ms=12.5, label='minimum volatility')
plt.plot(sigma_phi(max_sharpe['x']), mu_phi(max_sharpe['x']), 'v', ms=12.5, label='maximum Sharpe ratio')
plt.xlabel('expected volalitity')
plt.ylabel('expected return')
plt.legend()
plt.show()
