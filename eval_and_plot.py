# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt 
import numpy as np
import math
import random

# set a static seed for the RNG
random.seed(100)

# f and a arrays are used to contain the data
f = []
a = []

# Using an 80/20 split in training and testing dataset with 17 data points
# yields an n of 13. Ncn stands for N choose n.
N = 17
n = 13
Ncn = int(math.factorial(N)/(math.factorial(n) * math.factorial(N-n)))

# initialize R matrix and row vector v
R = np.zeros((Ncn, N))
v = np.zeros(N)
ci = 0

# gen_vec() function - recursively fills the R matrix with the N choose n
#                      possibilities.
#
# args: v - row vector
#       n - the same n as used in N choose n
#       idx - row index in the R matrix    
def gen_vec(v, n, idx):
    global ci
    sv = sum_v(v)
    if sv == n - 1:
        while idx < N:
            if v[idx] == 0:
                v[idx] = 1
                R[ci, ...] = v
                ci += 1
                v[idx] = 0
            idx += 1
    else:
        while idx < N:
            if v[idx] == 0:
                v[idx] = 1
                gen_vec(v, n, idx + 1)
                v[idx] = 0
            idx += 1

# sum_v() function - simple function to calculate the sum of a vector v
#
# args: v - vector to iterate through and calculate sum.
def sum_v(v):
    s = 0
    for e in v:
        s += e
    return s

# this function fills the R matrix by recursively enumerating the N choose
# n possibilities
gen_vec(v, n, 0)

# fill the f and a arrays from the data file.
with open('D:\school\ELG5255\project\code\data.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    
    for row in plots: 
        if float(row[0]) >= 100 and float(row[0]) <= 900:
            f.append(float(row[0])) 
            a.append(float(row[1])) 

# -----------------------------------------------------------------------------
# model 1: step function
# i: partitions
i = 0

# E: error variable
E = 0 

# k, h: iterators
k = 0
h = 0

# f_idx: frequency index for the previous frequency in a
f_idx = 0

# p1, p2: two pseudo-random iterations seeded statically to identify
# example partitions
p1 = int(random.random() * Ncn)
p2 = int(random.random() * Ncn)

# train_f, train_a: training data
train_f = []
train_a = []

# test_f, test_a: test data
test_f = []
test_a = []

# step_idx: the index of the previous f and a
step_idx = 0

# plot_func_g and plot_func_b: plotting data
plot_func_g = np.empty(10000)
plot_func_b = np.empty(10000)

# for each partition
while i < Ncn:
    # if we're in one of the highlighted partition, plot them.
    if i == p1 or i == p2:
        k = 0
        # for each N, sort the data into training and testing
        while k < N:
            # add data points either training or testing set depending on R
            if R[i, k] == 1:
                train_f.append(f[k])
                train_a.append(a[k])
            else:
                test_f.append(f[k])
                test_a.append(a[k])
            k += 1
        # plot the resulting two data sets as well as the function.
        k = 0
        while (k < 10000):
            plot_func_g[k] = k/10.0
            if step_idx > n - 2:
                break;
            if plot_func_g[k] > train_f[step_idx+1]:
                step_idx = step_idx + 1
            plot_func_b[k] = train_a[step_idx]
            k += 1
        
        while k < 10000:
            plot_func_g[k] = k/10.0
            plot_func_b[k] = train_a[step_idx]
            k += 1
        
        print(R[i, :])
        plt.scatter(train_f, train_a, c='blue') 
        plt.scatter(test_f, test_a, c='green')
        plt.plot(plot_func_g, plot_func_b, c='red')
        plt.legend(['Training set', 'Test set', 'Approximation function'])
        plt.xlabel('Frequency (MHz)') 
        plt.ylabel('Cable Loss (dB/100m)') 
        plt.title('Step function') 
        plt.show()
        
        # clean the data so that on the next iteration to plot, it starts fresh
        plt.clf()
        plot_func_g = np.empty(10000)
        step_idx = 0
        plot_func_b = np.empty(10000)
        train_f = []
        train_a = []
        test_f = []
        test_a = []
    # evaluating error: for each frequency k in the test set
    k = 0
    while k < N:
        if R[i,k] == 0:
            # find the appropriate frequency h in the training set
            h = 0
            f_idx = 0
            while h < N:
                if f[h] <= R[i,h]*f[k]:
                    f_idx = h
                h += 1
            # evaluate the error between the two and add it to E
            E += (a[f_idx] - a[k]) ** 2
        k += 1
    i += 1
E /= Ncn
print('expected error, step function: ', E)

# -----------------------------------------------------------------------------
# model 2: local linear approximation
# i: partitions
i = 0

# E: error variable
E = 0 

# k, h: iterators
k = 0
h = 0

# f_idx: frequency index for the previous frequency in a
f_idx = 0

# f_idx2: frequency index for the next frequency in a
f_idx2 = 0

# w and w0: weights for the line between the two points
w = 0.0
w0 = 0.0

# p1, p2: two pseudo-random iterations seeded statically to identify
# example partitions
p1 = int(random.random() * Ncn)
p2 = int(random.random() * Ncn)

# train_f, train_a: training data
train_f = []
train_a = []

# test_f, test_a: test data
test_f = []
test_a = []

# step_idx: the index of the previous f and a
step_idx = 0

# plot_func_g and plot_func_b: plotting data
plot_func_g = np.empty(10000)
plot_func_b = np.empty(10000)

# for each partition
while i < Ncn:
    # if we're in one of the highlighted partition, plot them.
    if i == p1 or i == p2:
        k = 0
        # for each N, sort the data into training and testing
        while k < N:
            # add data points either training or testing set depending on R
            if R[i, k] == 1:
                train_f.append(f[k])
                train_a.append(a[k])
            else:
                test_f.append(f[k])
                test_a.append(a[k])
            k += 1
        
        # plot the resulting two data sets as well as the function.
        # starting from 0 until reaching the first training frequency,
        # fill the plot function y value with the first frequency
        k = 0
        while k/10.0 <= train_f[0]:
            plot_func_g[k] = k/10.0
            plot_func_b[k] = train_a[0]
            k += 1
        
        # between the first and last frequencies, use the appropriate
        # line between two points to calculate the approximation.
        while k/10 < train_f[n-1]:
            plot_func_g[k] = k/10.0
            if plot_func_g[k] > train_f[step_idx]:
                w = (train_a[step_idx]-train_a[step_idx+1])/(train_f[step_idx]-train_f[step_idx+1])
                w0 = train_a[step_idx] - w*train_f[step_idx]
                step_idx += 1
            plot_func_b[k] = w*plot_func_g[k]+w0
            k += 1
        
        # at the end, use the last value of the function.
        while k < 10000:
            plot_func_g[k] = k/10.0
            plot_func_b[k] = train_a[step_idx]
            k += 1
        
        print(R[i, :])
        plt.scatter(train_f, train_a, c='blue') 
        plt.scatter(test_f, test_a, c='green')
        plt.plot(plot_func_g, plot_func_b, c='red')
        plt.legend(['Training set', 'Test set', 'Approximation function'])
        plt.xlabel('Frequency (MHz)') 
        plt.ylabel('Cable Loss (dB/100m)') 
        plt.title('Local Linear Approximation') 
        plt.show()
        plt.clf()
        
        # clean the data so that on the next iteration to plot, it starts fresh
        plot_func_g = np.empty(10000)
        step_idx = 0
        plot_func_b = np.empty(10000)
        train_f = []
        train_a = []
        test_f = []
        test_a = []
    # for each frequency k in the test set
    k = 0
    while k < N:
        if R[i,k] == 0:
            # find the appropriate frequency h in the training set
            h = 0
            f_idx = 0
            f_idx2 = 0
            while h < N:
                if f[h] <= R[i,h]*f[k]:
                    f_idx = h
                h += 1
            h = N - 1
            while h > f_idx:
                if R[i, h] == 1:
                    f_idx2 = h
                h -= 1
            # construct the model
            w = (a[f_idx]-a[f_idx2])/(f[f_idx]-f[f_idx2])
            w0 = a[f_idx] - w*f[f_idx]
            # evaluate the error between the two
            E += ((w*f[k] + w0) - a[k]) ** 2
        k += 1
    i += 1
E /= Ncn
print('expected error, local linear approximation: ', E)

# -----------------------------------------------------------------------------
# model 3: simple linear regression
# i: partitions
i = 0

# E: error variable
E = 0 

# k, h: iterators
k = 0

# sa, sf, saf, sf2: summation variables. they are used to calculate the
# regression parameters.
sa = 0
sf = 0
saf = 0
sf2 = 0

# w and w0: regression parameters.
w = 0.0
w0 = 0.0

# p1, p2: two pseudo-random iterations seeded statically to identify
# example partitions
p1 = int(random.random() * Ncn)
p2 = int(random.random() * Ncn)

# train_f, train_a: training data
train_f = []
train_a = []

# test_f, test_a: test data
test_f = []
test_a = []

# step_idx: the index of the previous f and a
step_idx = 0

# plot_func_g and plot_func_b: plotting data
plot_func_g = np.empty(10000)
plot_func_b = np.empty(10000)

# for each partition i
while i < Ncn:
    # construct the model based on the analytic solution
    k = 0
    sa = 0
    sf = 0
    saf = 0
    sf2 = 0
    while k < N:
        sa += a[k]*R[i,k]
        sf += f[k]*R[i,k]
        saf += f[k]*a[k]*R[i,k]
        sf2 += (f[k]**2)*R[i,k]
        k += 1
    w = (sa*sf/n - saf)/((sf**2)/n - sf2)
    w0 = (sa-w*sf)/n
    
    # plot the data if it's iteration p1 or p2
    if i == p1 or i == p2:
        k = 0
        # for each N
        while k < N:
            # add data points either training or testing set depending on R
            if R[i, k] == 1:
                train_f.append(f[k])
                train_a.append(a[k])
            else:
                test_f.append(f[k])
                test_a.append(a[k])
            k += 1
        
        # fill the plotting x and y values based on the regression result
        k = 0
        while k < 10000:
            plot_func_g[k] = k/10.0
            plot_func_b[k] = w*plot_func_g[k] + w0
            k += 1
        
        # plot 
        print(R[i, :])
        plt.scatter(train_f, train_a, c='blue') 
        plt.scatter(test_f, test_a, c='green')
        plt.plot(plot_func_g, plot_func_b, c='red')
        plt.legend(['Training set', 'Test set', 'Approximation function'])
        plt.xlabel('Frequency (MHz)') 
        plt.ylabel('Cable Loss (dB/100m)') 
        plt.title('Linear Regression') 
        plt.show()
        plt.clf()
        
        # reset variables
        plot_func_g = np.empty(10000)
        plot_func_b = np.empty(10000)
        train_f = []
        train_a = []
        test_f = []
        test_a = []
        
    # for each frequency k in the test set
    k = 0
    while k < N:
        if R[i,k] == 0:
            # evaluate the error between the two
            E += ((w*f[k] + w0) - a[k]) ** 2
        k += 1
    i += 1
E /= Ncn
print('expected error, simple linear regression: ', E)

# -----------------------------------------------------------------------------
# model 4: linear regression log input gradient descent
# i: partitions
i = 0

# E: error variable
E = 0 

# k, h: iterators
k = 0

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

# init_eta: inital value for eta
init_eta = 0.01

# delta: delta for decreasing eta at each iteration.
delta = init_eta/(max_ep+1)

# c: parameter to optimize through gradient descent.
c = 2.0

# p1, p2: two pseudo-random iterations seeded statically to identify
# example partitions
p1 = int(random.random() * Ncn)
p2 = int(random.random() * Ncn)

# train_f, train_a: training data
train_f = []
train_a = []

# test_f, test_a: test data
test_f = []
test_a = []

# step_idx: the index of the previous f and a
step_idx = 0

# plot_func_g and plot_func_b: plotting data
plot_func_g = np.empty(10000)
plot_func_b = np.empty(10000)

# for each partition i
while i < Ncn:
    # construct the model
    ep = 0
    eta = init_eta
    # perform the gradient descent with max_ep epochs
    while ep < max_ep:
        sa = 0
        sf = 0
        saf = 0
        sf2 = 0
        sc = 0
        
        k = 0
        while k < N:
            sa += (c**a[k])*R[i,k]
            sf += f[k]*R[i,k]
            saf += f[k]*(c**a[k])*R[i,k]
            sf2 += (f[k]**2)*R[i,k]
            k += 1
            
        w = (sa*sf/n - saf)/((sf**2)/n - sf2)
        w0 = (sa-w*sf)/n
        
        k = 0
        while k < N:
            sc += R[i,k]*a[k]*(c ** a[k])*(w*f[k] + w0 - c**a[k])
            k += 1
        
        c = c + eta * ((2/c)*sc)
        eta = eta - delta
        ep += 1
    
    # optimal w, w0 and c values have been obtained. if this iteration is
    # either p1 or p2, plot the data.
    if i == p1 or i == p2:
        k = 0
        # for each N
        while k < N:
            # add data points either training or testing set depending on R
            if R[i, k] == 1:
                train_f.append(f[k])
                train_a.append(a[k])
            else:
                test_f.append(f[k])
                test_a.append(a[k])
            k += 1
        
        # fill the plotting x and y values based on optimal w, w0 and c values
        k = 0
        while k < 10000:
            plot_func_g[k] = k/10.0
            plot_func_b[k] = math.log(w*plot_func_g[k] + w0)/math.log(c)
            k += 1
        
        # perform the plot
        print(R[i, :])
        plt.scatter(train_f, train_a, c='blue') 
        plt.scatter(test_f, test_a, c='green')
        plt.plot(plot_func_g, plot_func_b, c='red')
        plt.legend(['Training set', 'Test set', 'Approximation function'])
        plt.xlabel('Frequency (MHz)') 
        plt.ylabel('Cable Loss (dB/100m)') 
        plt.title('Linear Regression on transformed data') 
        plt.show()
        plt.clf()
        
        # clean the storage variables
        plot_func_g = np.empty(10000)
        plot_func_b = np.empty(10000)
        train_f = []
        train_a = []
        test_f = []
        test_a = []
    
    # for each frequency k in the test set
    k = 0
    while k < N:
        if R[i,k] == 0:
            # evaluate the error between the two
            E += (math.log(w*f[k] + w0)/math.log(c) - a[k]) ** 2
        k += 1
    i += 1
E /= Ncn
print('expected error, linear regression log input gradient descent: ', E)




