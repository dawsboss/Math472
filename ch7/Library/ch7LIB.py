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
def GaussianPT1(A,B, verbose=False):
    a = np.concatenate((A, B), axis=1)
    n = a.shape[0]
    for i in range(n):
        if verbose == True:
            print(a)
            print()
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
def GaussianPT2(A,B, verbose=False):
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
def GausElim(A,B,verbose=False):
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




######
##  7.4 code
######



#7.4 - Alg 7.5 - LU decomposition algorithm (no pivoting)
#   Parameters:
#       A - numpy matrix - NxN - Square
#   Output:
#       LUFact - numpy matrix 
#       indx - list of indexes of -----
def factorization(a):
    """
    Perform Decomposition
    """
    n = a.shape[0]
    A = a.copy()
    for i in range(n):
        for j in range(i+1, n):
            A[j, i] = A[j, i]/A[i, i]
            for k in range(i+1, n):
                A[j, k] = A[j, k] - A[j, i]*A[i, k]
    return A


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

#7.4 - Not from book - Splits the books LU output into it's L and U matricies
#   Parameters:
#       A - numpy matrix - NxN - Square
#   Output:
#       L - numpy matrix - botton left triangle
#       U - numpy matrix -top right triangle
def LUExplode(A):
    rtn = False
    L = np.copy(A)
    U = np.copy(A)
    (Arows, Acols) = np.shape(A)
    if( Arows == Acols ):
        L = np.tril(L,-1)
        np.fill_diagona(L,1)
        U = np.triu(U)
        rtn = (L,U)
    else:
        print("Matrix is not square!")
    return rtn
                
#7.4 - Not from book - Take a matrix and returns it's L and U with its permutation
#   Parameters:
#       A - numpy matrix - NxN - Square
#   Output:
#       L - numpy matrix - botton left triangle
#       U - numpy matrix - top right triangle                
#       P - numpy matrix - permutation matrix (Stuff gets shifted around from the pivioting)
#   P@L@U == A
def LUExplodePerm(A, idx):
    rtn = False
    (Arows, Acols) = np.shape(A)
    if( Arows == Acols ):
        (L,U) = LUExplode(A)
        P = np.zeros((Arows,Acols))
        for i in range(len(idx)):
            P[idx[i],i] = 1
        rtn = (P,L,U)
    else:
        print("Matrix is not square")
    return rtn()

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
                



######
##  7.5 code
######


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
    return math.pow(np.sum(np.pow(v,2)),1.0/2.0)
        
# Calculate P-norm of a vector
def vectorPNorm(v, p):
    return math.pow(np.sum(np.power(np.abs(v),p)), 1.0/p)


### Advanced alg

#7.5 - Def 7.3 - Condition Number
#   Parameters:
#       A - numpy matrix 
#       Norm - function pointer ot a norm function
#   Output:
#       k(A) - Condition Number
def conditionNumber(A, Norm):
    Ainv = np.linalg.inv(A)
    return Norm(A) * Norm(Ainv)

#7.5 - Thrm 7.3 - Condition Number
#   Parameters:
#       A - numpy matrix 
#   Output:
#       k(A) - Condition Number
def conditionNumberEsitimation(A):
    (row, col) = np.shape(A)
    alpha = matrixInfinityNorm(A)
    
    (X,idx) = LU(A)
    
    y = np.random.rand(row,1)#This can be cahnge TODO
    for i in range(5):
        y = y/vectorInfinityNorm(y)
        y = LUsolver(X,y,idx)
    
    return alpha * vectorInfinityNorm(y)

#7.5 - Def 7.4 - Gaussian Elimination Growth Factor
#   Parameters:
#       A - numpy matrix
#   Output:
#       
def growthFactor(A_):
    rtn = False
    A = np.copy(A_)
    (row, col)=np.shape(A)
    gn = -1
    if(row == col):
        for i in range(row-1):
            for j in range(i+1, row):
                m = A[j,i]/A[i,i]
                for k in range(row):
                    A[j,k] = A[j,k] - m*A[i,k]
            gn = max(np.max(np.abs(A)),gn)
        N = matrixInfinityNorm(A_)
        rtn = gn/N
    else: 
        print("Inputted matrix must be square!")
    return rtn


#Used for testing the library code
if __name__ == '__main__':
    B = HnMatrix(5)
    print(B)
    print("I am stray lib code")
