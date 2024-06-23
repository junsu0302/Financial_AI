import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from data import f_test, f_train, l_test, l_train

def MSE(l:np.ndarray, p:np.ndarray) -> np.ndarray:
  """
  평균 제곱 오차(MSE) 계산

  Args:
    l (np.ndarray): 실제 값
    p (np.ndarray): 예측 값

  Returns:
    np.ndarray: 평균 제곱 오차
  """
  return np.mean((l - p) ** 2)

# 모델 평가 함수
def evaluate(reg:np.ndarray, f:np.ndarray, l:np.ndarray):
  """
  모델을 평가하고 평가 결과를 출력

  Args:
    reg (np.ndarray): 다항 회귀 모델의 계수
    f (np.ndarray): 특징 데이터
    l (np.ndarray): 실제 값
  """
  p = np.polyval(reg, f)
  bias = np.abs(l - p).mean()
  var = p.var()
  msg = f'MSE={MSE(l, p):.4f} | R2={r2_score(l, p):.4f} | bias={bias:.4f} | var={var:.4f}'
  print(msg)

reg = {}
mse = {}
for d in range(1, 22, 4):
  reg[d] = np.polyfit(f_train, l_train, deg=d)
    
  # 훈련 데이터에 대한 예측
  p = np.polyval(reg[d], f_train)
  mse_train = MSE(l_train, p)
    
  # 테스트 데이터에 대한 예측
  p = np.polyval(reg[d], f_test)
  mse_test = MSE(l_test, p)
    
  # MSE 저장
  mse[d] = (mse_train, mse_test)
    
  # 모델 평가
  evaluate(reg[d], f_train, l_train)

fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 훈련 데이터 시각화
ax[0].plot(f_train, l_train, 'ro', label='Training data')
ax[0].set_title('Training Data and Polynomial Fits')
ax[0].set_xlabel('Features')
ax[0].set_ylabel('Labels')

# 테스트 데이터 시각화
ax[1].plot(f_test, l_test, 'go', label='Testing data')
ax[1].set_title('Testing Data and Polynomial Fits')
ax[1].set_xlabel('Features')
ax[1].set_ylabel('Labels')

for d in reg:
  # 훈련 데이터에 대한 다항 회귀 결과 시각화
  p = np.polyval(reg[d], f_train)
  ax[0].plot(f_train, p, '--', label=f'deg={d} train')
    
  # 테스트 데이터에 대한 다항 회귀 결과 시각화
  p = np.polyval(reg[d], f_test)
  ax[1].plot(f_test, p, '--', label=f'deg={d} test')

plt.tight_layout()
ax[0].legend()
ax[1].legend()
plt.savefig('assets/ML/regression.png')
plt.show()