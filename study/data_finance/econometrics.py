import numpy as np

def f(x):
  return 2 + 1 / 2 * x

x = np.arange(-4, 5)
y = f(x)

beta = np.cov(x, y, ddof=0)[0, 1] / x.var()
alpha = y.mean() - beta * x.mean()

y_ = alpha + beta * x

print(np.allclose(y, y_))