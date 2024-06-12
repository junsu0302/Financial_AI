import numpy as np

#? 주식 및 채권 가격 설정 (0: 현재, 1: 만기)
S0 = 10
S1 = np.array((20, 5))
B0 = 10
B1 = np.array((11, 11))

#? 자산 벡터 설정 (0: 현재, 1: 만기)
M0 = np.array((S0, B0))
M1 = np.array((S1, B1)).T

#? 행사가 설정
K = 14.5

#? 만기 시점에서의 콜 옵션 가치 계산
C1 = np.maximum(S1 - K, 0) # [5.5, 0.]

#? 옵션 포트폴리오 구성 계산
phi = np.linalg.solve(M1, C1) # [0.36666667 -0.16666667]

#? 검증 : 포트폴리오가 옵션의 페이오프를 재현하는지 검증
np.allclose(C1, np.dot(M1, phi)) # True

#? 현재 시점에서의 콜 옵션 가치 계산
C0 = np.dot(M0, phi)