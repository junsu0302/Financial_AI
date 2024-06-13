import numpy as np
from scipy.optimize import minimize

#? 주식 및 채권 가격 설정 (0: 현재, 1: 만기)
S0 = 10
S1 = np.array([20, 5])
B0 = 10
B1 = np.array([11, 11])

#? 자산 벡터 설정 (0: 현재, 1: 만기)
M0 = np.array([S0, B0])
M1 = np.array([S1, B1]).T

#? 행사가 설정
K = 14.5

#? 초기 포트폴리오 배분 설정
phi_S = np.array([0.75, 0.25])
phi_B = np.array([0.25, 0.75])

#? 만기 포트폴리오 가치 계산
final_S = np.dot(M1, phi_S)
final_B = np.dot(M1, phi_B)

#? 시나리오 확률 설정
P = np.array([0.5, 0.5])

#? 효용 함수
def u(x):
  return np.sqrt(x)

#? 기대 효용 함수
def EUT(phi):
  portfolio_values = np.dot(M1, phi)
  return np.dot(P, u(portfolio_values))

#? 리스크 (포트폴리오 분산) 계산 함수
def portfolio_variance(phi):
  portfolio_values = np.dot(M1, phi)
  mean_portfolio_value = np.dot(P, portfolio_values)
  variance = np.dot(P, (portfolio_values - mean_portfolio_value) ** 2)
  return variance

#? 초기 자산 가치
initial_wealth = 10

#? 제약조건
constraints = [
  {
    'type': 'eq',
    'fun': lambda phi: np.dot(M0, phi) - initial_wealth
  },
  {
    'type': 'ineq',
    'fun': lambda phi: max_risk - portfolio_variance(phi)
  }
]

#? 최대 허용 리스크 설정
max_risk = 10

#? 최적화
optimization_result = minimize(lambda phi: -EUT(phi),
                               x0=phi_S,
                               constraints=constraints)

#? 결과 출력
print("Optimization result:")
print(optimization_result)

optimal_phi = optimization_result['x']
optimal_EUT = EUT(optimal_phi)
print("Optimal expected utility:", optimal_EUT)

optimal_risk = portfolio_variance(optimal_phi)
print("Optimal portfolio variance (risk):", optimal_risk)
