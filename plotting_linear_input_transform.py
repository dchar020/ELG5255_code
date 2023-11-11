# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt 
import numpy as np
import math

# f and a are the data storage variables
f = []
a = []

# g and b are the approximation variables
g = np.empty(10000)
b = np.empty(10000)

# k, i: iterator
k = 0
i = 0

# sa, sf, saf, sf2: summation variables. they are used to calculate the
# regression parameters.
sa = 0
sf = 0
saf = 0
sf2 = 0

# w and w0: regression parameters.
w = 0.0
w0 = 0.0

# ep: gradient descent iterator - epoch
ep = 0

# max_ep: max epoch - max value it can take.
max_ep = 500

# eta: gradient descent parameter
eta = 0.01

# c: parameter to optimize through gradient descent.
c = 2.0

# len_f: number of data points
len_f = 0

# e: total error
e = 0

with open('D:\school\ELG5255\project\code\data.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    
    for row in plots: 
        f.append(float(row[0])) 
        a.append(float(row[1])) 
    
    len_f = len(f)
    
    # for each epoch,
    while ep < max_ep:
        sa = 0
        sf = 0
        saf = 0
        sf2 = 0
        sc = 0
        
        # calculate optimal parameters
        i = 0
        while i < len_f:
            sa += c ** a[i]
            sf += f[i]
            saf += (c ** a[i])*f[i]
            sf2 += f[i] ** 2
            i += 1
            
        w = (sa*sf/len_f - saf)/((sf ** 2)/len_f - sf2)
        w0 = sa/len_f - w*sf/len_f
        
        i = 0
        while i < len_f:
            sc += a[i]*(c ** a[i])*(w*f[i] + w0 - c**a[i])
            i += 1
        
        # update c using gradient descent
        c = c + eta * ((2/c)*sc)
        ep += 1
    
    # generate plotting data
    i = 0
    while i < 10000:
        g[i] = i/10.0
        b[i] = w*g[i] + w0
        i += 1
    
    # modify the input data to linearize
    i = 0
    while i < len_f:
        a[i] = c ** a[i]
        i += 1
    
    # calculate error
    i = 0
    while i < len_f:
        e += (w*f[i]+w0 -a[i]) ** 2
        i += 1
    
    # show optimal parameters, error and plot the data
    print(e, w, w0, c)
    
    plt.scatter(f, a) 
    plt.plot(g, b, c='red')
    plt.legend(['c^a(f)', 'b(g) - Linear Regression on c^a(f)'])
    plt.xlabel('Frequency (MHz)') 
    plt.ylabel('Value') 
    plt.title('Linear Regression on the Optimal c Value') 
    plt.show() 
    
    
    