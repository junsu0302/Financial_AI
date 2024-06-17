import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.optimize import minimize

from portfolio import rM, mu_phi, sigma_phi, frontier

init_printing(use_unicode=False, use_latex=False)

# TODO: 자본 시장선

#? 환경 설정
phi_M = np.array((0.8, 0.2))  # 시장 포트폴리오 배분
mu_M = mu_phi(phi_M)          # 시장 포트폴리오 기대 수익률 
sigma_M = sigma_phi(phi_M)    # 시장 포트폴리오 변동성
r = 0.0025                    # 무위험 이자율

#? 자본 시장선 점들 계산
cml_vol = np.linspace(0, 0.6, 100)                      # 변동성 범위를 0 ~ 0.6까지 100개의 값으로 생성
cml_ret = r + ((mu_M - r) / sigma_M) * cml_vol          # 자본 시장선의 수익률 계산
capital_market_line = np.vstack((cml_vol, cml_ret)).T   # 변동성과 수익률을 결합하여 자본 시장선 행렬 생성

# plt.figure(figsize=(10, 6))
# plt.plot(frontier[:, 0], frontier[:, 1], 'm.', ms=5, label='efficient frontier')
# plt.plot(0, r, 'o', ms=9, label='risk-less asset')
# plt.plot(sigma_M, mu_M, '^', ms=9, label='market portfolio')
# plt.plot((0, 0.6), (r, r + ((mu_M - r) / sigma_M) * 0.6), 'r', label='capital market line', lw=2.0)
# plt.annotate('$(0, \\bar{r})$', (0, r), (-0.015, r + 0.01))
# plt.annotate('$(\sigma_M, \mu_M)$', (sigma_M, mu_M), (sigma_M-0.025, mu_M+0.01))
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.legend()
# plt.show()

# TODO: 무차별 곡선
mu, sigma, b, v = symbols('mu sigma b v')
sol = solve('mu - b / 2 * (sigma ** 2 + mu ** 2) - v', mu)

u1 = sol[0].subs({'b': 1, 'v': 0.1})
u2 = sol[0].subs({'b': 1, 'v': 0.125})

f1 = lambdify(sigma, u1)
f2 = lambdify(sigma, u2)

sigma_ = np.linspace(0.0, 0.5)
u1_ = f1(sigma_)
u2_ = f2(sigma_)

# plt.figure(figsize=(10, 6))
# plt.plot(sigma_, u1_, label='$v=0.1$')
# plt.plot(sigma_, u2_, '--', label='$v=0.125$')
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.legend()
# plt.show()

# TODO: 최적 포트폴리오

def U(p):
  mu, sigma = p
  return mu - 1 / 2 * (sigma ** 2 + mu ** 2)

cons = {
  'type': 'eq',
  'fun': lambda p: p[0] - (r + (mu_M - r) / sigma_M * p[1])
}

opt = minimize(lambda p: -U(p), (0.1, 0.3), constraints=cons)

u = sol[0].subs({'b': 1, 'v': -opt['fun']})

f = lambdify(sigma, u)

u_ = f(sigma_)

plt.figure(figsize=(10, 6))
plt.plot(0, r, 'o', ms=9, label='rist-less asset')
plt.plot(sigma_M, mu_M, '^', ms=9, label='market portfolio')
plt.plot(opt['x'][1], opt['x'][0], 'v', ms=9, label='optimal portfolio')
plt.plot((0, 0.5), (r, r + (mu_M - r) / sigma_M * 0.5), label='capital market line', lw=2.0)
plt.plot(sigma_, u_, '--', label='$v={}$'.format(-round(opt['fun'], 3)))
plt.annotate(f'$(0, \\bar{{r}}={r:.4f})$', (0, r), xytext=(+0.02, r))
plt.annotate(f'$(\\sigma_M={sigma_M:.4f}, \\mu_M={mu_M:.4f})$', (sigma_M, mu_M), xytext=(sigma_M-0.01, mu_M-0.01))
plt.annotate(f'$(\\sigma_{{opt}}={opt["x"][1]:.4f}, \\mu_{{opt}}={opt["x"][0]:.4f})$', (opt['x'][1], opt['x'][0]), xytext=(opt['x'][1] - 0.01, opt['x'][0] - 0.01))
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.legend()
plt.show()