#!/usr/bin/env python3

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import math

def create_dataset(name, init_size=10, max_size=100, step=1, fill_value=1, ntests=1):
    h = max_size//step
    data = np.full((h+1, 1 + ntests), np.nan)
    pos = 0
    for i in range(init_size, max_size + step, step):
        data[pos, 0] = i
        for t in range(1, ntests + 1):
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

def sqr_error(data, mean):
    standart_deviation = np.empty(mean.size)
    data_matrix = data.values
    for i in range(1, len(standart_deviation)):
        standart_deviation[i] = (sum([(d - mean[i])**2 for d in data_matrix[i]])/(len(data_matrix[0]) - 1))**0.5

    return standart_deviation

s = pd.DataFrame()
# create_dataset('data.xlsx', init_size=0, max_size=1500, step=50, ntests=5)

df = pd.read_excel('data.xlsx')
# List of columns with time data
time = [t for t in df.keys() if 'Time' in t]
mean_by_row = df[time].mean(axis=1)
error = sqr_error(df[time], mean_by_row)

T = mean_by_row[1::]
N = df['Size'].values[1::]
log_N = np.log(N)
log_T = np.log(T)
k, b = np.polyfit(log_N, log_T, 1)


# line fitting explained here
# https://stackoverflow.com/questions/30657453/fitting-a-straight-line-to-a-log-log-curve-in-matplotlib
# plotting examples https://matplotlib.org/examples/pylab_examples/log_demo.html

# Initialize empty plot in log scale
plt.loglog()
# Line approximation
plt.loglog(N, np.exp(log_N*k + b))
plt.errorbar(N, T, error[1::], None, 'o', markersize=3)
plt.grid(True)
plt.show()
