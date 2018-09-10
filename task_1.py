#!/usr/bin/env python3

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import math
import os


def create_dataset_logspace(name, min_pwr=0, max_pwr=3.2, fill_value=1, ntests=1):
    pos = 0
    log_space = sorted({int(i) for i in np.logspace(min_pwr, max_pwr)} - {1})
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


def create_dataset_linspace(name, st, end, num, fill_value=1, ntests=1):
    pos = 0
    log_space = sorted({int(i) for i in np.linspace(st, end, num)} - {1})
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


PLOT_1_DATASET_NAME = 'plot_1_data.xlsx'
plot_1_dataset_params = {
                        'name': PLOT_1_DATASET_NAME,
                        'min_pwr': 0,
                        'max_pwr': 3.2,
                        'fill_value': 5,
                        'ntests': 3,
                        }
# Created dataset if it doesn't exist
if PLOT_1_DATASET_NAME not in os.listdir():
    create_dataset_logspace(**plot_1_dataset_params)

df = pd.read_excel(PLOT_1_DATASET_NAME)
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


PLOT_2_DATASET_NAME = 'plot_2_data.xlsx'
plot_2_dataset_params = {
                        'name': PLOT_2_DATASET_NAME,
                        'st': 2,
                        'end': 500,
                        'num': 400,
                        'fill_value': 5,
                        'ntests': 25,
                        }
# Created dataset if it doesn't exist
if PLOT_2_DATASET_NAME not in os.listdir():
    create_dataset_linspace(**plot_2_dataset_params)


df = pd.read_excel(PLOT_2_DATASET_NAME)

# List of columns with time data
time_cols = [t for t in df.keys() if 'Time' in t]
mean_by_row = df[time_cols].mean(axis=1)
error = df[time_cols].std(axis=1)

T = mean_by_row.values
N = df['Size'].values


# Filter time measurement outliers by relative error
rel_error = error / T * 100
REL_ERROR_THRESHOLD = 10 # %
error_mask = (rel_error < REL_ERROR_THRESHOLD) & (rel_error != 0)
T_inliers = T[error_mask]
N_inliers = N[error_mask]
error_inliers = error[error_mask]

# Shift values
N1 = np.array(N_inliers[:len(N_inliers) - 1]) 
N2 = np.array(N_inliers[1::])
T1 = np.array(T_inliers[:len(T_inliers) - 1])
T2 = np.array(T_inliers[1::])
T1_err = np.array(error_inliers[:len(error_inliers) - 1])
T2_err = np.array(error_inliers[1::])

# Calculate complexity 
n = np.abs(np.log(T2/T1)/np.log(N2/N1))
n_abs_err = np.log(T2_err/T1_err)
# n_abs_err = np.log((T2_err/T2)/(T1_err/T2))*n
n_rel_err = abs(n_abs_err/n*100)

# Set filtering parameters
mask = (n_rel_err < 20) & (n < 3.5) & (n > 2)
k, b = np.polyfit(N1[mask], n[mask], 1)

# Check log err bars https://faculty.washington.edu/stuve/log_error.pdf
# shall try calculate n using not mean T, but each column instead
plt.subplot(1,2,2)
plt.ylim((0, 4))
err, *__ = plt.errorbar(N1[mask], n[mask], yerr=n_abs_err[mask], fmt='o', markersize=2)
approx, *__ = plt.plot(N1[mask], N1[mask]*k + b, 'k-')
plt.xlabel('$N$')
plt.ylabel('$n(N)$')
plt.title('$Correlation$ $of$ $complexity$ $and$ $size$')
plt.legend(['$Approximation$', '$Mean$ $and$ $deviation$'], loc=4)
plt.grid(True)
plt.show()
