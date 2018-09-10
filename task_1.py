#!/usr/bin/env python3

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import math

def create_dataset_logspace(name, fill_value=1, ntests=1):
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

def create_dataset_linspace(name, fill_value=1, ntests=1):
    pos = 0
    log_space = sorted({int(i) for i in np.linspace(2, 500, 250)} - {1})
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

# create_dataset_logspace('data_3_tests.xlsx', ntests=3)

df = pd.read_excel('data_3_tests.xlsx')
# List of columns with time data
time_cols = [t for t in df.keys() if 'Time' in t]
mean_by_row = df[time_cols].mean(axis=1)
error = df[time_cols].std(axis=1)

T = mean_by_row.values
N = df['Size'].values
log_N = np.log(N)
log_T = np.log(T)
k, b = np.polyfit(log_N, log_T, 1)

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


# create_dataset_linspace('data_lin_5_tests.xlsx', ntests=5)

df = pd.read_excel('data_lin_5_tests.xlsx')
# List of columns with time data
time_cols = [t for t in df.keys() if 'Time' in t]
mean_by_row = df[time_cols].mean(axis=1)
error = df[time_cols].std(axis=1)

T = mean_by_row.values
N = df['Size'].values

rel_error = error / T * 100
REL_ERROR_THRESHOLD = 10 # %
non_zero = rel_error != 0
error_mask = rel_error < REL_ERROR_THRESHOLD
error_mask = error_mask & non_zero

# Filter obvious outliers
T_inliers = T[error_mask]
N_inliers = N[error_mask]
error_inliers = error[error_mask]


N2 = np.array(N_inliers[1::])
N1 = np.array(N_inliers[:len(N_inliers) - 1]) 
T2 = np.array(T_inliers[1::])
T1 = np.array(T_inliers[:len(T_inliers) - 1])
T2_err = np.array(error_inliers[1::])
T1_err = np.array(error_inliers[:len(error_inliers) - 1])

n = np.abs(np.log(T2/T1)/np.log(N2/N1))
n_err = np.exp(np.log(T2_err/T1_err))
n_rel_err = abs(n_err/n*100)

mask = (n_rel_err < 50) & (n < 4)
k, b = np.polyfit(N1[mask], n[mask], 1)


plt.subplot(1,2,2)
err, *__ = plt.errorbar(N1[mask], n[mask], yerr=n_err[mask], fmt='o', markersize=2)
approx, *__ = plt.plot(N1[mask], N1[mask]*k + b, 'k-')
plt.xlabel('$N$')
plt.ylabel('$n(N)$')
plt.title('$Correlation$ $of$ $complexity$ $and$ $size$')
plt.legend(['$Approximation$', '$Mean$ $and$ $deviation$'], loc=2)
plt.grid(True)
plt.show()
