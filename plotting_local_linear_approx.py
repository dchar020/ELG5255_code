# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt 
import numpy as np

# f and a are the data storage variables
f = []
a = []

# g and b are the approximation variables
g = np.empty(10000)
b = np.empty(10000)

# i: iterator
i = 0

# idx: index of the previous frequency
idx = 0

# w and w0: regression parameters.
w = 0.0
w0 = 0.0

# len_f: number of data points
len_f = 0

with open('D:\school\ELG5255\project\code\data.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    
    # read the data
    for row in plots: 
        f.append(float(row[0])) 
        a.append(float(row[1])) 
    
    len_f = len(f)
    
    # fill the output variable b with the output of the first frequency
    # until the first frequency is reached
    while i/10.0 <= f[0]:
        g[i] = i/10.0
        b[i] = a[0]
        i += 1
    
    # between the first and last frequencies, use the local linear 
    # approximation and generate outputs
    while i/10 < f[len_f-1]:
        g[i] = i/10.0
        if g[i] > f[idx]:
            w = (a[idx]-a[idx+1])/(f[idx]-f[idx+1])
            w0 = a[idx] - w*f[idx]
            idx += 1
        b[i] = w*g[i]+w0
        i += 1
    
    # for all the points after the last frequency, use the output of the last 
    # frequency
    while i < 10000:
        g[i] = i/10.0
        b[i] = a[idx]
        i += 1

    plt.scatter(f, a) 
    plt.plot(g, b, c='red')
    plt.legend(['a(f) - Data provided', 'b(g) - Local linear approximation'])
    plt.xlabel('Frequency (MHz)') 
    plt.ylabel('Cable Loss (dB/100m)') 
    plt.title('Cable Loss as a Function of Frequency') 
    plt.show() 
    
    
    