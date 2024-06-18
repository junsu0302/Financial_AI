import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve, lambdify
from scipy.optimize import minimize

#? 차익거래 가격결정 이론
def arbitrage_pricing(S0, S1, T0, T1, U0, U1, V1):
  M0 = np.array([S0, T0, U0])
  M1 = np.array([S1, T1, U1]).T

  reg = np.linalg.lstsq(M1, V1, rcond=-1)[0]
  V0 = np.dot(M0, reg)

  return V0, reg

#? 기대효용 이론
def expected_utility(S0, S1, T0, T1, utility_func, P):
  M0 = np.array((S0, T0))
  M1 = np.array((S1, T1)).T

  rM = M1 / M0 - 1
  phi = np.array((0.5, 0.5))

  def mu(rX):
    return np.dot(P, rX)

  def sigma(rX):
    return np.sqrt(np.mean((rX - mu(rX)) ** 2))

  def mu_phi(phi):
    return np.dot(phi, mu(rM))

  def sigma_phi(phi):
    return sigma(rM)

  def EUT(phi):
    x = np.dot(M1, phi)
    return np.dot(P, utility_func(x))

  w = 10
  cons = {'type': 'eq', 'fun': lambda phi: np.dot(M0, phi) - w}
  opt = minimize(lambda phi: -EUT(phi), x0=phi, constraints=cons)

  return opt, mu_phi, sigma_phi

#? 평균-분산 포트폴리오 이론
def mean_variance_portfolio(S0, S1, T0, T1, U0, U1, P):
  M0 = np.array([S0, T0, U0])
  M1 = np.array([S1, T1, U1]).T

  rM = M1 / M0 - 1

  def mu(rX):
    return np.dot(P, rX)

  def var(rX):
    return ((rX - mu(rX)) ** 2).mean()

  def sigma(rX):
    return np.sqrt(var(rX))

  def mu_phi(phi):
    return np.dot(phi, mu(rM))

  def var_phi(phi):
    cv = np.cov(rM.T, aweights=P, ddof=0)
    return np.dot(phi, np.dot(cv, phi))

  def sigma_phi(phi):
    return var_phi(phi) ** 0.5

  phi_mcs = np.random.random((3, 200))
  phi_mcs = (phi_mcs / phi_mcs.sum(axis=0)).T
  mcs = np.array([(sigma_phi(phi), mu_phi(phi)) for phi in phi_mcs])

  return mcs, mu_phi, sigma_phi

#? 자본자산 가격결정 모델
def capm(S0, S1, T0, T1, rf, phi_M):
  M0 = np.array((S0, T0))
  M1 = np.array((S1, T1)).T

  rM = M1 / M0 - 1

  def mu(rX):
    return np.dot(P, rX)

  def sigma(rX):
    return np.sqrt(np.mean((rX - mu(rX)) ** 2))

  def mu_phi(phi):
    return np.dot(phi, mu(rM))

  def sigma_phi(phi):
    return sigma(rM)

  mu_M = mu_phi(phi_M)
  sigma_M = sigma_phi(phi_M)

  cml_vol = np.linspace(0, 0.6, 100)
  cml_ret = rf + ((mu_M - rf) / sigma_M) * cml_vol
  capital_market_line = np.vstack((cml_vol, cml_ret)).T

  return capital_market_line

#? 불확실성과 리스크
def risk_uncertainty(mu, sigma, b, v):
  sol = solve('mu - b / 2 * (sigma ** 2 + mu ** 2) - v', mu)
  u = sol[0].subs({'b': b, 'v': v})
  f = lambdify(sigma, u)

  sigma_ = np.linspace(0.0, 0.5)
  u_ = f(sigma_)

  return sigma_, u_

if __name__ == "__main__":
  S0, S1 = 10, np.array((20, 10, 5))
  T0, T1 = 10, np.array((1, 12, 13))
  U0, U1 = 10, np.array((12, 5, 11))
  V1 = np.array((12, 15, 7))
  rf = 0.0025
  phi_M = np.array((0.8, 0.2))
  P = np.ones(3) / 3

  #? 차익거래 가격결정 이론
  V0, reg = arbitrage_pricing(S0, S1, T0, T1, U0, U1, V1)
  print('='*26, "Arbitrage Pricing", '='*26)
  print("Current Price V0: ", V0)
  print("Regression Coefficients: ", reg)
  print('='*72)

  #? 기대효용 이론
  utility_func = np.sqrt
  opt, mu_phi, sigma_phi = expected_utility(S0, S1, T0, T1, utility_func, P)
  print('='*27, "Expected Utility", '='*27)
  print("Optimal Portfolio: ", opt)
  print('='*72)

  #? 평균-분산 포트폴리오 이론
  mcs, mu_phi, sigma_phi = mean_variance_portfolio(S0, S1, T0, T1, U0, U1, P)

  #? 자본자산 가격결정 모델
  capital_market_line = capm(S0, S1, T0, T1, rf, phi_M)

  #? 불확실성과 리스크
  mu, sigma, b, v = symbols('mu sigma b v')
  sigma_, u_ = risk_uncertainty(mu, sigma, 1, 0.1)

  #? 모든 그래프를 하나의 플롯에 추가
  plt.figure(figsize=(14, 10))

  #? 평균-분산 포트폴리오 이론 그래프
  plt.plot(mcs[:, 0], mcs[:, 1], 'ro', label='Mean-Variance Portfolio')

  #? 자본자산 가격결정 모델 그래프
  plt.plot(capital_market_line[:, 0], capital_market_line[:, 1], 'b-', label='Capital Market Line')

  #? 불확실성과 리스크 그래프
  plt.plot(sigma_, u_, 'g--', label='Risk and Uncertainty')

  plt.xlabel('Expected Volatility')
  plt.ylabel('Expected Return')
  plt.title('Combined Financial Theories')
  plt.legend()
  plt.show()
