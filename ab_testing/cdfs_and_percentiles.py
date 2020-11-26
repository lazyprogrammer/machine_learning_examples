import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


mu = 170
sd = 7


# generate samples from our distribution
x = norm.rvs(loc=mu, scale=sd, size=100)

# maximum likelihood mean
x.mean()

# maximum likelihood variance
x.var()

# maximum likelihood std
x.std()

# unbiased variance
x.var(ddof=1)

# unbiased std
x.std(ddof=1)

# at what height are you in the 95th percentile?
norm.ppf(0.95, loc=mu, scale=sd)

# you are 160 cm tall, what percentile are you in?
norm.cdf(160, loc=mu, scale=sd)

# you are 180 cm tall, what is the probability that someone is taller than you?
1 - norm.cdf(180, loc=mu, scale=sd)