#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:42:06 2021

@author: grant
"""


import math
import numpy as np
import matplotlib.pyplot as plt

def KnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = 2
            elif abs(i-j)==1:
                rtn[i,j]= -1
            else:
                rtn[i,j] = 0
    return rtn

def TnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = 4
            elif abs(i-j)==1:
                rtn[i,j] = 1
            else:
                rtn[i,j] = 0
    return rtn

def LU(A):
    rtn = False
    a = np.copy(A)
    (Arows, Acols) = np.shape(a)
    if(Arows == Acols):
        indx = list(range(Arows))
        for i in range(Arows-1):
            #Pivoting
            am = abs(a[i,i])
            p = i
            for j in range(i+1, Arows):
                if(abs(a[j,i] > am)):
                    am = abs(a[j,i])
                    p = j
            if(p > i):
                for k in range(Arows):
                    hold = a[i,k].copy()
                    a[i,k] = a[p,k].copy()
                    a[p,k] = hold.copy()
                ihold = indx[i]
                indx[i] = indx[p]
                indx[p] = ihold
            #Elimination 
            for j in range(i+1, Arows):
                a[j,i] = a[j,i]/a[i,i]
                for k in range(i+1, Arows):
                    a[j,k] = a[j,k] - a[j,i] * a[i,k]
        rtn = [ a, indx ]
    else:
        print("Matrix is not square!")
    return rtn

def LUsolver(X, bvec, ivec):
    b = np.copy(bvec)
    rows,cols = np.shape(X)
    x = np.zeros((rows, 1))
    for k in range(rows):
        x[k] = b[ivec[k]][0]
    for k in range(rows):
        b[k] = x[k]
    y = [ b[0][0] ]
    for i in range(1, rows):
        s=0.0
        for j in range(i):
            s = s +X[i,j] * y[j]
        y.append(b[i][0] - s)
    x[rows-1]=y[rows-1]/X[rows-1,rows-1]
    for i in range(rows-2, -1, -1):
        s = 0.0
        for j in range(i+1, rows):
            s = s + X[i,j] * x[j]
        x[i] = (y[i]-s)/X[i,i]
    return x



A = np.matmul(KnMatrix(10),TnMatrix(10))
B = np.matrix([[-4],[-1],[0],[0],[0],[0],[0],[0],[10],[40]], dtype=float)

(LUfact, idx) = LU(A)
x = LUsolver(LUfact, B, idx)
print("x:\n",x)