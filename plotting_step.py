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

with open('D:\school\ELG5255\project\code\data.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    
    # read the data
    for row in plots: 
        f.append(float(row[0])) 
        a.append(float(row[1])) 
    
    # calculate the step function variable based on the index (idx) of the
    # previous frequency encountered.
    while i < 10000:
        g[i] = i/10.0
        if idx > len(f) - 2:
            break;
        if g[i] > f[idx+1]:
            idx = idx + 1
        b[i] = a[idx]
        i += 1
    
    # for the remaining x values, use the last frequency's output.
    while i < 10000:
        g[i] = i/10.0
        b[i] = a[idx]
        i += 1

    plt.scatter(f, a) 
    plt.plot(g, b, c='red')
    plt.legend(['a(f) - Data provided', 'b(g) - Step function'])
    plt.xlabel('Frequency (MHz)') 
    plt.ylabel('Cable Loss (dB/100m)') 
    plt.title('Cable Loss as a Function of Frequency') 
    plt.show() 
    
    
    