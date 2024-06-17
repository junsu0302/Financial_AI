import numpy as np

#? 금융 상품 가격 설정 (0: 현재, 1: 만기)
S0, S1 = 10, np.array((20, 10, 5))
T0, T1 = 10, np.array((1, 12, 13))
U0, U1 = 10, np.array((12, 5, 11))

#? 자산 벡터 설정 (0: 현재, 1: 만기)
M0 = np.array([S0, T0, U0])
M1 = np.array([S1, T1, U1]).T

#? 목표 상품의 만기 가격
V1 = np.array((12, 15, 7))

#? 요인 로딩 행렬 B (각 자산이 각 요인에 대해 얼마나 민감한지)
B = np.array([
  [0.8, 0.2],
  [0.6, 0.3],
  [0.4, 0.4]
])

#? 요인의 현재 가격 및 만기 가격 (예시 데이터)
F0 = np.array([10, 10])
F1 = np.array([
  [12, 15],
  [13, 14],
  [14, 13]
])

#? 회귀 분석을 통해 요인 벡터 f_t 계산
f_t = np.linalg.lstsq(F1, V1, rcond=-1)[0]

#? 상수항 벡터 a 계산
a = V1 - np.dot(B, f_t)

#? 목표 상품의 현재 가격
V0 = a + np.dot(B, f_t)

print("Regression coefficients (f_t): ", f_t)
print("Intercept vector (a): ", a)
print("Estimated current price V0: ", V0)