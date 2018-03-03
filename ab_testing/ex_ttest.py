# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import pandas as pd
from scipy import stats

# get data
df = pd.read_csv('advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action']
b = b['action']

print("a.mean:", a.mean())
print("b.mean:", b.mean())

# built-in t-test:
t, p = stats.ttest_ind(a, b)
print("t:\t", t, "p:\t", p)

# welch's t-test:
t, p = stats.ttest_ind(a, b, equal_var=False)
print("Welch's t-test:")
print("t:\t", t, "p:\t", p)

# welch's t-test manual:
N1 = len(a)
s1_sq = a.var()
N2 = len(b)
s2_sq = b.var()
t = (a.mean() - b.mean()) / np.sqrt(s1_sq / N1 + s2_sq / N2)

nu1 = N1 - 1
nu2 = N2 - 1
df = (s1_sq / N1 + s2_sq / N2)**2 / ( (s1_sq*s1_sq) / (N1*N1 * nu1) + (s2_sq*s2_sq) / (N2*N2 * nu2) )
p = (1 - stats.t.cdf(np.abs(t), df=df))*2
print("Manual Welch t-test")
print("t:\t", t, "p:\t", p)