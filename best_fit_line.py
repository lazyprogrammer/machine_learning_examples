from pulp import *

### remove variable b because it is unconstrained
### it's just a linear combination of the others
### you can get the result:
# status: Optimal
# values:
#     a: 20000.0
#     b: -10000.0
#     c: 0.0
#     z: 10000.0
### or any other multiple thereof
### ax + by - c = 0
### is the same as y = (-a/b)x + (c/b)


prob = LpProblem("best_fit_line", LpMinimize)
z = LpVariable('z',0)
a = LpVariable('a',0)
# b = LpVariable('b')
c = LpVariable('c',0)

# objective function
prob += z

points = [
    (1,3),
    (2,5),
    (3,7),
    (5,11),
    (7,14),
    (8,15),
    (10,19),
]

prob += (a != 0)
for x,y in points:
    prob += (a*x - y - c <= z)
    prob += (a*x - y - c >= -z)

status = prob.solve(GLPK(msg = 0))
print "status:", LpStatus[status]
print "values:"
print "\ta:", value(a)
# print "\tb:", value(b)
print "\tc:", value(c)
print "\tz:", value(z)


# extra part to plot everything
import numpy as np
import matplotlib.pyplot as plt
data = np.array(points)
plt.scatter(data[:,0], data[:,1])
x = np.linspace(0, 11, 100)
y = value(a)*x - value(c)
plt.plot(x, y)
plt.show()
