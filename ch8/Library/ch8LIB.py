#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:26:09 2021

@author: grant
"""

import numpy as np
import math
import sympy as sy
from sympy.utilities.lambdify import lambdify, implemented_function


### Basic alg

# Calculate the Infinity Norm of a matrix (row)
def matrixInfinityNorm(A_):
    A = np.copy(np.abs(A_))
    rowsum = np.sum(A,axis=1).tolist()
    return np.max(rowsum)

# Calculate the Infinity Min of a vector
def vectorInfinityMin(v):
    return np.min(np.abs(v))

# Calculate the Infinity Norm of a vector
def vectorInfinityNorm(v):
    return np.max(np.abs(v))

# Calculate 2-Norm of a matrix
def matrix2Norm(A_):
    A = np.matmul(A_, np.transpose(A_))
    (eigval, eigvect) = np.linalg.eig(A)
    return math.sqrt(np.max(eigval))
    

# Calculate 2-Norm of a vector
def vector2Norm(v):
    return math.sqrt(np.sum(np.square(v)))
        
# Calculate P-norm of a vector
def vectorPNorm(v, p):
    return math.pow(np.sum(np.power(np.abs(v),p)), 1.0/p)

#Hessenberg parsing helper:
#   Parameters: 
#       A - square numpy matrix
#   Outputs:
#       a11
#       a
#       bT
#       A22
def HesParse(A_, i):
    (row, col) = np.shape(A_)
    if(row!=col):#Matrix needs to be square
        return False
    A = A_.copy()
    a11=A[i,i]
    a=A[i+1:col, i]
    bT=A[i, i+1:col]
    A22=A[i+1:col, i+1:col]
    return(a11, a, bT, A22)

#Hessenberg creator itterative:
#   Parameters: 
#       A - square numpy matrix
#   Outputs:
#       Ah - square numpy matrix in hessenberg's form
def HessenbergItter(A_, verbose=False):
    (row, col) = np.shape(A_)
    if(row!=col):#Matrix needs to be square
        return False
    A = A_.copy()
    p = np.eye(row,col)
    for i in range(0, row-2):#From col 0 to col n-2
        a11, a, bT, A22 = HesParse(A, i)#parsing matrix
        if(a[0,0] > 0):
            c=vector2Norm(a)
        else:
            c=-1 * vector2Norm(a)
        w = np.add(a, (c*np.eye(row-(i+1), 1, 0, dtype=float)))
        gamma = 2/math.pow(vector2Norm(w),2)
        Q = np.eye(row-(i+1),col-(i+1), dtype=float) - (gamma* np.matmul(w, w.T))
        P = np.eye(row,col, dtype=float)
        P[-Q.shape[0]:, -Q.shape[1]:] = Q
        PAP = np.matmul(P, np.matmul(A,P.T))
        p = np.matmul(p,P)
        A[-PAP.shape[0]:, -PAP.shape[1]:] = PAP
        if(verbose):
            print(f"i: {i}\na11: {a11}\na: {a}\nbT:{bT}\nA22{A22}\nc: {c}\nw: {w}\ngamma: {gamma}\nQ: {Q}\nP: {P}\nPAP: {PAP}\nA: {A}\n\n\n")
        #print(A)
    return A,p
    
    
#Used for testing the library code
if __name__ == '__main__':
    import numpy as np
    import math
    import ch7.Library.ch7LIB as ch7
    import sympy as sy
    from sympy.utilities.lambdify import lambdify, implemented_function
    from sympy import Lambda 
    import inspect
    
    # A = np.matrix([[6,2,1,1],
    #                [2,6,2,1],
    #                [1,2,6,2],
    #                [1,1,2,6]],dtype=float)
    A = np.matrix([[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12],
                   [13,14,15,16]],dtype=float)    
    a11,a,b,B = HesParse(A,0)
    print(a11)
    print()
    print(a)
    print()
    print(b)
    print()  
    print(B)
    print() 
    print(np.eye(5, 1, 0))
    print()
    B=np.eye(3,3)
    A[-B.shape[0]:, -B.shape[1]:] = B
    print(A)
    print()
    A = np.matrix([[6,2,1,1],
                    [2,6,2,1],
                    [1,2,6,2],
                    [1,1,2,6]],dtype=float)  
    X = HessenbergItter(A)
    print(X)
    
