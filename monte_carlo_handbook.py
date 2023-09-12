#!/usr/bin/env python
# coding: utf-8

# ## Monte Carlo Handbook  
# _An Introduction to the Principles of Monte Carlo Simulation_
# 
# This markdown file should serve as an educational resource to assist in the learning of Monte Carlo simulation methods
# 
# The goal here is to provide clear and comprehensive explanations of relevant concepts
# 
# An effort is made it keep language used as non-technical and intuitive as possible
# 
# Additions of clear explanations, useful examples and coded visualisations are encouraged 

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# _Introduction_
# 
# The idea behind Monte Carlo simulation is relatively simple;
# 
# Take a unit square with a smaller circle of unknown dimensions inside. Say we want to find an estimate of the area circle without directly measuring. How can we do this?
# 
# Lets randomly drop points on the unit square and count the number that fall inside of the circle. As we increase the number of points dropped, the **proportion of points** inside the circle will be approximately equal to the area of the circle.
# 
# Example;
# 
# Take the following function
# 
# \begin{equation}
# f:[0,1]\to[0,1],\ f(x)=1-\exp\{\sin^2(\pi x)\}\cos^2(\pi x).
# \end{equation}
# 
# Suppose we want an approximation of the following integral
# 
# \begin{equation}
# \int_0^1 f(x) \mathrm dx. \label{integral}
# \end{equation}
# 
# If we generate points on $[0,1]^2$ and calculate the emprical probability of a point falling below the curve, this will approximate the area under the curve as well i.e. the integral we are looking for
# 
# More formally we want to generate n points $(x_i,y_i)$  and find the proportion of points that satisfy the following relation, $y_i\leq f(x_i)$
# 
# 

# In[11]:


def my_func(x):
    return 1 - (math.e**(math.sin(math.pi*x)**2))*math.cos(math.pi*x)**2

x = np.random.uniform(0, 1, 10**4)
y = np.random.uniform(0, 1, 10**4)

fx = [my_func(i) for i in x]
print(f'Our estimate is {sum([i < j for i,j in zip(y, fx)])/10**4}')
print(f'Actual integral is {integrate.quad(my_func, 0, 1)[0]}')


# Using $n = 10^4$ samples, we get a pretty close approximation
# 
# The first graph below shows how the accuracy of the approximation imoproves as $n$ gets large. The second eclearly shows the mechanism used.

# In[14]:


def my_func_2(n):
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    
    fx = [my_func(i) for i in x]
    return sum([i<j for i,j in zip(y, fx)])/n


n = np.arange(0, 10**4, 10)
fx = [my_func_2(i) for i in n]

plt.plot(n, fx, 'm')
plt.show()

rng = np.arange(0,1, 1/150)
out = [my_func(i) for i in rng]

x = np.random.uniform(0, 1, 250)
y = np.random.uniform(0, 1, 250)

tf = [True if y[i] <= my_func(x[i])  else False for i in range(0,250)]
neg_tf = [not i for i in tf]
plt.plot(rng, out, 'k')
plt.scatter(x[tf], y[tf], c='g')
plt.scatter(x[neg_tf], y[neg_tf], c='r')

plt.show()


# In[ ]:




