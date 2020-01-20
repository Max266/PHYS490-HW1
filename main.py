#!/usr/bin/python

import numpy as np
from numpy.linalg import inv
import random
import json
import os
import sys

DataQ1 = open(sys.argv[1], "r")
Datajson = open(sys.argv[2], "r")

## Analytic

lines = DataQ1.readlines()
x=[]
y=[]

for i in lines:
    y.append(float(i.split(' ')[-1]))
    m = []
    j = 0
    while (float(i.split(' ')[j])) != (float(i.split(' ')[-1])):
        m.append(float(i.split(' ')[j]))
        j += 1
    x.append([1] + m)

x = np.array(x)
y = np.array(y)

xT = x.transpose()

w_analytic = (inv(xT.dot(x)).dot(xT)).dot(y)

## Stochastic Gradient Descent

learning_data = json.load(Datajson)

learning_rate = learning_data['learning rate']
num_iter = learning_data['num iter']

xs = x.shape[1]

w_gd = np.array([[1]]*xs)
w_gd = np.matrix(w_gd)


k = 0
while k < num_iter:
    phi = random.randint(0, xs-1)
    x_new = np.matrix(x[phi]).transpose()
    y_new = np.matrix(y[phi])
    w_gd = w_gd + learning_rate * int((y_new - w_gd.transpose()*x_new)[0]) * x_new
    k += 1


## Create Out File

w_gd = list(map(lambda x: format(round(x[0],4), '.4f'), w_gd.tolist()))
w_analytic = list(map(lambda x: format(round(x,4), '.4f'), w_analytic.tolist()))

file_out_name = (os.path.splitext(sys.argv[1])[0]) + ".out"
file_out = open(file_out_name,"w+")

Answer = w_analytic + ['\n'] + w_gd

for l in Answer:
    file_out.write(l + '\n')


file_out.close()


