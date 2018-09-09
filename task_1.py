#!/usr/bin/env python3

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import math

def create_dataset(name, fill_value=1, ntests=1):
    pos = 0
    log_space = sorted({int(i) for i in np.logspace(0, 3.2)} - {1})
    data = np.full((len(log_space)+1, 1 + ntests), np.nan)
    for i in log_space:
        data[pos, 0] = i
        for t in range(1, ntests + 1):
            print('Size', i, 'Test:', t)
            m = np.full((i, i), fill_value)
            start_time = time()
            m = np.dot(m, m)
            result = time() - start_time
            data[pos, t] = result
        pos += 1
    header = ['Size'] + ['Time ' + str(i) for i in range(1, ntests + 1)]
    df = pd.DataFrame(data, columns=header)
    df = df.dropna()
    with pd.ExcelWriter (name) as xlsx:
        df.to_excel(xlsx)

# create_dataset('data.xlsx', ntests=7)

df = pd.read_excel('data.xlsx')
# List of columns with time data
time = [t for t in df.keys() if 'Time' in t]
mean_by_row = df[time].mean(axis=1)
error = df[time].std(axis=1)

T = mean_by_row
N = df['Size'].values
log_N = np.log(N)
log_T = np.log(T)
k, b = np.polyfit(log_N, log_T, 1)
print('n =', k)

# line fitting explained here
# https://stackoverflow.com/questions/30657453/fitting-a-straight-line-to-a-log-log-curve-in-matplotlib
# plotting examples https://matplotlib.org/examples/pylab_examples/log_demo.html

# Initialize empty figure with fixated size
plt.figure(figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# subplot(n-rows, n-columns, position)
plt.subplot(1,2,1)

# Line approximation
approx, *__ = plt.loglog(N, np.exp(log_N*k + b), 'k-')
err, *__ = plt.errorbar(N, T, yerr=error, fmt='o', markersize=2)

plt.xlabel('$log(N)$')
plt.ylabel('$log(T)$')
plt.title('$Multiplication$ $time$ $from$ $matrix$ $size$')
plt.legend(['$Approximation$', '$Mean$ $and$ $deviation$'])
plt.grid(True)


plt.subplot(1,2,2)


plt.xlabel('$N$')
plt.ylabel('$n(N)$')
plt.title('$Correlation$ $of$ $complexity$ $and$ $size$')
plt.show()
