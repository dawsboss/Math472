#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:24:10 2021

@author: grant
Library for chapter 7
"""
import numpy as np
import math

#Hn Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Hn-style Matrix
def HnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            rtn[i,j] = 1.0/( i+j+1 )
    return rtn

#Kn Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Kn-style Matrix
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

#Tn Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Tn-style Matrix
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

#An Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy An-style Matrix
def AnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = 1
            elif i-j==1:
                rtn[i,j] = 4
            elif i-j==-1:
                rtn[i,j] = -4
            else:
                rtn[i,j] = 0
    return rtn


#7.2 - Alg 7.1 - Native Gaussian Elimination 
#   Parameters:
#       A - numpy matrix - NxN - Square
#       B - numpy matrix - 1xN - Vector
#   Output:
#       numpy matrix that is in echelon form
def GaussianPT1(A,B):
    a = np.concatenate((A, B), axis=1)
    n = a.shape[0]
    for i in range(n):
        if a[i,i] == 0.0:
            print("Dividing by zero!")
            pass
             
        for j in range(i+1, n):
            ratio = a[j,i]/a[i,i]
             
            for k in range(n+1):
                a[j,k] = a[j,k] - ratio * a[i,k]
    b = a[:,n]
    a = np.delete(a, n, 1)
    return (a, b)

#7.2 - Alg 7.2 - Backward Solution 
#   Parameters:
#       A - numpy matrix - NxN - Square
#       B - numpy matrix - 1xN - Vector
#   Output:
#       Vector X - Solution from Ax=B
def GaussianPT2(A,B):
    n = A.shape[0]
    X = np.zeros((n,1))
    X[n-1] = B[n-1] / A[n-1,n-1]
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i, n):
            sum = sum + A[i,j]*X[j]
        X[i] = (B[i]-sum)/A[i,i]
    return X

#7.2 - Alg 7.4 - Gaussian Elimination with Partial Pivoting
#   Parameters:
#       A - numpy matrix - NxN - Square
#       B - numpy matrix - Nx1 - Vector
#   Output:
#       numpy matrix that is in echelon form
def GausElim(A,B):
    a = np.concatenate((A, B), axis=1)
    
    n = A.shape[0]
    for i in range(0, n):
        am = abs(a[i,i])
        p=i
        for j in range(i+1, n):
            if abs(a[j,i]) > am:
                am = abs(a[j,i])
                p=j
    #print(B)
        if p > i:
            for k in range(i, n+1):
                hold = a[i,k]
                a[i,k] = a[p,k]
                a[p,k] = hold
        for j in range(i+1, n):
            ratio = a[j,i]/a[i,i]
             
            for k in range(n+1):
                a[j,k] = a[j,k] - ratio * a[i,k]
    
    b = a[:,n]
    a = np.delete(a, n, 1)
    return (a,b)
                



#Used for testing the library code
if __name__ == '__main__':
    B = HnMatrix(5)
    print(B)
    print("I am stray lib code")
