import matplotlib.pyplot as plt

lr = 1e-2
x1 = 5
x2 = -5

def J(x1, x2):
  return x1**2 + x2**4

def g1(x1):
  return 2*x1

def g2(x2):
  return 4*x2**3

values = []
for i in range(1000):
  values.append(J(x1, x2))
  x1 -= lr * g1(x1)
  x2 -= lr * g2(x2)
values.append(J(x1, x2))

print(x1, x2)
plt.plot(values)
plt.show()