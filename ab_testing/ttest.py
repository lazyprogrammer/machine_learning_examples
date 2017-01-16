# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
import numpy as np
from scipy import stats

# generate data
N = 10
a = np.random.randn(N) + 2 # mean 2, variance 1
b = np.random.randn(N) # mean 0, variance 1

# roll your own t-test:
var_a = a.var(ddof=1) # unbiased estimator, divide by N-1 instead of N
var_b = b.var(ddof=1)
s = np.sqrt( (var_a + var_b) / 2 ) # balanced standard deviation
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N)) # t-statistic
df = 2*N - 2 # degrees of freedom
p = 1 - stats.t.cdf(np.abs(t), df=df) # one-sided test p-value
print "t:\t", t, "p:\t", 2*p # two-sided test p-value

# built-in t-test:
t2, p2 = stats.ttest_ind(a, b)
print "t2:\t", t2, "p2:\t", p2