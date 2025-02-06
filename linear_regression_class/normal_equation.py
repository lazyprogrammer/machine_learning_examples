#!/usr/bin/env python
# coding: utf-8

# In[42]:


#Importing all the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import numpy as np
from numpy.linalg import inv
import random


# In[45]:


''' According to normal equation method the optimal value of parameter theta for which cost function is minimum can be computed by 
 setting the derivative of cost function with respect to each value of theta in the parameter matrix to 0
 By this method we get the parameter matrix theta equal to:
 theta = inverse(X'*X)*X'*Y   , where X' = transpose of X(feature matrix) and where  Y is the target matrix'''
if __name__== '__main__':
    df=pd.read_csv('file1.txt',names=['x','y'])                     #the file has been added to the repository
    x=df['x']
    y=df['y']
    m=x.shape[0]                                                    #no. of training examples
    X=[np.array([x[i]])for i in range (m)]                          #converting the table of input features into a matrix
    Y=[np.array([y[i]]) for i in range (m)]                         #converting the target features values into a matrix
    Xtrans=np.transpose(X)
    theta_best=np.dot((np.dot((inv(np.dot(Xtrans,X))),Xtrans)),y)   #computing the value of theta for which cost function is minimum by using the normal equation
    
    
    #plot to evaluate perfromance
    for i in range (x.shape[0]):
        y_predict=theta_best*X
    
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()

