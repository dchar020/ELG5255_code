# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt 
import numpy as np

f = []
a = []

with open('D:\school\ELG5255\project\code\data.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    
    # read the data
    for row in plots: 
        f.append(float(row[0])) 
        a.append(float(row[1])) 

    # plot it
    plt.scatter(f, a) 
    plt.xlabel('Frequency (MHz)') 
    plt.ylabel('Cable Loss (dB/100m)') 
    plt.title('Cable Loss as a Function of Frequency') 
    plt.show() 
    
    
    