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
import numpy.linalg as la

### Basic alg - CH 7 ###

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

#7.4 - Alg 7.7 - LU Decomposition with Partial Pivoting
#   Parameters:
#       A - numpy matrix - NxN - Square
#   Output:
#       LUFact - numpy matrix 
#       indx - list of indexes of -----
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

#7.4 - ALg 7.8 - Forward-Backward Solution, Using LU Decomposition
#   Parameters:
#       X - numpy matrix - NxN - Square
#       bvec - numpy matrix - Nx1
#       ivec - numpy matrix - Nx1
#   Output:
#       x - numpy matrix - Ax=B, this is the x
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



###### CH 8 ######

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

#Basic Power Method:
#   Parameters: 
#       A - square numpy matrix
#       tol - Error tollerance for estimation
#   Outputs:
#       lambda - estimated largest lambda
def BasicPowerMethod(A_,tol,zk=False):
    A=np.copy(A_)
    (row, col) = np.shape(A)
    prevMu = 0
    mu=0
    i=0
    err = 10**12
    if(zk == False):
        zk = np.random.rand(row,1)
    
    while(err >= tol):
        yk = np.matmul(A,zk)
        mu = yk[np.argmax(abs(yk)),0]
        prevMu = mu
        err = abs(mu - prevMu)
        zk = yk/mu
        i+=1
    return mu, zk, i

#Inverse Power Method:
#   Parameters: 
#       A - square numpy matrix
#       err - Error tollerance for estimation
#       zk - (OPTIONAL) init guess vector
#   Outputs:
#       lambda - estimated smallest lambda
def InversePowerMethod(A, err, zk=False):
    (row, col) = np.shape(A)
    prevMu = 0
    mu=0
    i=0
    if(zk == False):
        zk = np.random.rand(row,1)
    
    while(i==0 or err < abs(mu-prevMu)):
        #yk,_,_ = ch7.LU()
        #(LUfact, idx) = LU(A)
        #yk = LUsolver(LUfact, zk, idx)
        yk = np.linalg.solve(A, zk)
        
        mu = yk[np.argmax(abs(yk)),0]
        prevMu = mu
        zk = yk/mu
        i+=1
    return 1/mu, zk, i

#QR factorization:
#   Parameters: 
#       A - square numpy matrix
#   Outputs:
#       Q,R - two square matricies the same size of A that
#               when multiplied otgether make A
def get_QR(A_, verbose=False):
    m,n = A_.shape
    Q = np.eye(n)
    R = A_.copy()
    for i in range(n-1):
        vector = A_[i:, i]
        e1 = np.zeros((vector.shape[0],1))
        e1[0] = 1
        u = vector2Norm(vector)*e1
        if vector[0] < 0:
            u=-u
        omega = vector + u
        omega = omega/vector2Norm(omega)
        H = np.eye(n)
        H[i:, i:] -= (2*(omega@omega.T))
        R = H@R
        Q = Q@H.T
    return Q,R

#Eigen values:
#   Parameters: 
#       A - square numpy matrix
#   Outputs:
#       eigVal - vector of eigen values
def get_eigvals(A__, include_imag=False):
    A = A__.copy()
    
    Q, R = get_QR(A)

    A_=R@Q
    
    prev = A_
    itt = 0
    for i in range(40):
        Q_, R_ = get_QR(A_)
        prev = A_
        itt += 1
        A_ = R_@Q_
        #print(f"{A_}\n")
    #print(itt)
    #if include_imag:
    #    return sorted([e for e in la.eigvals(A)])
    return sorted([e.real for e in la.eigvals(A)])#sorted([e for e in la.eigvals(A) if np.isreal(e)])
#np.diag(A_), sorted([e for e in la.eigvals(A)])
    
    #if( not include_imag):
    #    return sorted([e for e in la.eigvals(A)])
    #else:
    #    return sorted([float(e) for e in la.eigvals(A) if np.isreal(e)])
    
#Eigen values:
#   Parameters: 
#       A - vector of polynomials 
#   Outputs:
#       eigVal - vector of eigen values
def get_companion(A_, rotated=False):
    A = A_.copy()
    n,m = A.shape
    use = 0
    if n>m: 
        use=n-1
    else:
        use=m-1
    A11 = np.eye(use-1)
    #print(A11)
    rtn = np.zeros((use,use))
    #print(rtn)
    #print(f"{A11.shape[1]} {A11.shape[0]}")
    #print(rtn[-A11.shape[0]:, -A11.shape[1]:])
    rtn[-A11.shape[0]:, :A11.shape[1]] = A11
    #print(rtn[:, use-1:])
    
    if n>m:    
        A.reshape(1,use+1)
        #print(f"A: {A[:, 0:use]}")
        rtn[:, use-1:] = -A[:, :use-1]
    else:
        A.reshape(use+1,1)
        #print(A.T[:use])
        rtn[:, use-1:] = -A.T[:use]
    return rtn

    
    
    
    
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
    p = np.matrix([-5, -1, 2, 0, 1])
    C1 = get_companion(p, rotated=False)
    print(C1)
    x = get_eigvals(C1)
    #print(f" x[0]: {x[0]}\n\n x[1]: {x[1]}\n\n x[2]: {x[2]}")
    
