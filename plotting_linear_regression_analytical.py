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

# i: iterator
i = 0

# w and w0: regression parameters.
w = 0.0
w0 = 0.0

# len_f: number of data points
len_f = 0

# sa, sf, saf, sf2: summation variables. they are used to calculate the
# regression parameters.
sa = 0
sf = 0
saf = 0
sf2 = 0

# e: total error
e = 0

with open('D:\school\ELG5255\project\code\data.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    
    for row in plots: 
        f.append(float(row[0])) 
        a.append(float(row[1])) 
    
    len_f = len(f)
    
    # generate optimal w and w0
    while i < len_f:
        sa += a[i]
        sf += f[i]
        saf += a[i]*f[i]
        sf2 += f[i] ** 2
        i += 1
        
    w = (sa*sf/len_f - saf)/((sf ** 2)/len_f - sf2)
    w0 = sa/len_f - w*sf/len_f
    
    # generate data and store it in the b and g arrays
    i = 0
    while i < 10000:
        g[i] = i/10.0
        b[i] = w*g[i] + w0
        i += 1
    
    # calculate error
    i = 0
    while i < len_f:
        e += (w*f[i]+w0 -a[i]) ** 2
        i += 1
    print(e, w, w0)
    
    plt.scatter(f, a) 
    plt.plot(g, b, c='red')
    plt.legend(['a(f) - Data provided', 'b(g) - Linear Regression'])
    plt.xlabel('Frequency (MHz)') 
    plt.ylabel('Cable Loss (dB/100m)') 
    plt.title('Cable Loss as a Function of Frequency') 
    plt.show() 
    
    
    