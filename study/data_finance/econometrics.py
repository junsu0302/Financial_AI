import numpy as np

def f(x):
  return 2 + 1 / 2 * x

x = np.arange(-4, 5) # [-4, -3, -2, -1, 0, 1, 2, 3, 4]
y = f(x) # [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]

beta = np.cov(x, y, ddof=0)[0, 1] / x.var()
alpha = y.mean() - beta * x.mean()

y_ = alpha + beta * x

print(np.allclose(y, y_))