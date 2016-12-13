#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:41:23 2016

@author: Trismeg
"""

import numpy as np
import matplotlib.pyplot as plt

N=400
D=2

X=np.random.randn(N,D)

X[:200,:] = X[:200,:] - 2*np.ones((200,D))
X[200:,:] = X[200:,:] + 2*np.ones((200,D))
X[:100,0]=-X[:100,0]
X[200:300,0]=-X[200:300,0]


T=np.array([0]*100 + [1]*100 + [0]*100 + [1]*100)
 
ones=np.array([np.ones(N)]).T
              
plt.scatter(X[:,0],X[:,1],c=T,s=100, alpha=0.5)
plt.show()

xy=np.array([X[:,0]*X[:,1]]).T
Xb=np.concatenate((ones,xy,X),axis=1)

w=np.random.randn(D+2)

z=Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
Y=sigmoid(z)
    
def cross_entropy(T,Y):
    E=0
    for i in range(N):
        if T[i]==1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learning_rate=0.001
error=[]
for i in range(5000):
    e=cross_entropy(T,Y)
    error.append(e)
    if i%100==0:
        print(e)
    w += learning_rate*(np.dot((T-Y).T,Xb)-0.01*w)
    Y=sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross entropy")
print("Final w:",w)
print("Final classification rate:" ,1-np.abs(T-np.round(Y)).sum()/N)
