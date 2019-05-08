# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y

j = np.load('ddpg_results.npz')

returns = j['train']
test_returns = j['test']
q_losses = j['q_losses']
mu_losses = j['mu_losses']

plt.plot(returns)
plt.plot(smooth(np.array(returns)))
plt.title("Train returns")
plt.show()

plt.plot(test_returns)
plt.plot(smooth(np.array(test_returns)))
plt.title("Test returns")
plt.show()

plt.plot(q_losses)
plt.title('q_losses')
plt.show()

plt.plot(mu_losses)
plt.title('mu_losses')
plt.show()