#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:24:10 2021

@author: grant
Library for chapter 7
"""
import numpy as np
import math
import sympy as sy
from sympy.utilities.lambdify import lambdify, implemented_function

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
                rtn[i,j] = 2.0
            elif abs(i-j)==1:
                rtn[i,j]= -1.0
            else:
                rtn[i,j] = 0.0
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
                rtn[i,j] = 4.0
            elif abs(i-j)==1:
                rtn[i,j] = 1.0
            else:
                rtn[i,j] = 0.0
    return rtn

#Tn Matrix creator function: produces same as Tn matrix but the 4's are negative
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Tn-style Matrix
def TnNegMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = -4.0
            elif abs(i-j)==1:
                rtn[i,j] = 1.0
            else:
                rtn[i,j] = 0.0
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
                rtn[i,j] = 1.0
            elif i-j==1:
                rtn[i,j] = 4.0
            elif i-j==-1:
                rtn[i,j] = -4.0
            else:
                rtn[i,j] = 0.0
    return rtn


#7.2 - Alg 7.1 - Native Gaussian Elimination 
#   Parameters:
#       A - numpy matrix - NxN - Square
#       B - numpy matrix - 1xN - Vector
#   Output:
#       numpy matrix that is in echelon form
def GaussianPT1(A_,B_, verbose=False):
    A=np.copy(A_)
    B=np.copy(B_)
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
def GaussianPT2(A_,B_, verbose=False):
    A=np.copy(A_)
    B=np.copy(B_)
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
def GausElim(A_,B_,verbose=False):
    A = np.copy(A_)
    b = np.copy(B_)
    n = A.shape[0]
    for i in range(n-1):
        am = abs(A[i,i])
        p = i
        for j in range(i+1, n):
            if abs(A[j,i]) > am:
                am = abs(A[j,i])
                p = j
        if p > i:
            for k in range(i, n):
                hold = A[i,k]
                A[i,k] = A[p,k]
                A[p,k] = hold
            hold = np.copy(b[i])
            b[i] = np.copy(b[p])
            b[p] = hold
        for j in range(i+1, n):
            m = A[j,i]/A[i,i]
            for k in range(i+1, n):
                A[j,k] = A[j,k] - m*A[i,k]
            b[j] = b[j] - m*b[i]
    
    """
    n = A.shape[0]
    x = np.zeros(n)
    x[n - 1] = b[n - 1]/A[n - 1, n - 1]
    for i in range(n-1, 0, -1):
        sum_ = 0
        for j in range(i+1, n):
            sum_ = sum_ + A[i,j]*x[j]
        x[i] = (b[i] - sum_)/A[i,i]
    """
    return A, b




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
    return math.sqrt(np.sum(np.square(v)))
        
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
#      growth number
      
def growthFactor(A_):
    rtn = False
    A = np.copy(A_)
    (row, col)=np.shape(A)
    temp = -1
    if(row == col):
        for i in range(row-1):
            for j in range(i+1, row):
                m = A[j,i]/A[i,i]
                for k in range(row):
                    A[j,k] = A[j,k] - m*A[i,k]
            temp = max(np.max(np.abs(A)),temp)
        N = matrixInfinityNorm(A_)
        rtn = temp/N
    else: 
        print("Inputted matrix must be square!")
    return rtn

#7.6 - Def 7.6 - LDL Factorization
#   Parameters:
#       A - numpy matrix
#   Output:
#       L - numpy matrix
#       D - numpy matrix
def LDL(A_):
    A=np.copy(A_)
    row,col = np.shape(A)
    L = np.eye(row)
    D = np.zeros((row,col))
    
    #2x2 base case
    D[0,0] = A[0,0].copy()
    for i in range(1, row):
        L[i,0] = A[i,0]/A[0,0]
    D[1,1] = A[1,1] - (A[0,1]**2)/A[0,0]
    
    #Check if we need more than 2x2
    if(row>2):
        for i in range(2,row):
            for j in range(1,i):
                L[i,j] = A[i,j]
                for k in range(j):
                    L[i,j] = L[i,j] - L[i,k]*L[j,k]*D[k,k]
                L[i,j] = L[i,j]/D[j,j]
            D[i,i]=A[i,i]
            for k in range(i):
                D[i,i] = D[i,i] - (L[i,k]**2)*D[k,k]
        return (L,D)

########
##7.7
########

#7.7 -  - Jacobi Iteration
#   Parameters:
#       A - numpy matrix
#       b - numpy vector(nx1)
#       x0 - single number to start itteration at
#       n - number of itterations
#   Output:
#       x - numpy matrix (nx1)
def JacobiIteration(A, b, x0, n, verbose=False):
    rtn = x0.copy()
    Diag = np.diagonal(A)
    (row, col) = np.shape(A)
    D = np.zeros(shape=(row,col))
    for i in range(row):
        D[i,i] = 1/Diag[i];
    I = np.eye(row)
    negDA = np.subtract(I, np.matmul(D,A))
    DB = np.matmul(D,b)
    for i in range(n):
        rtn = np.add(np.matmul(negDA, rtn), DB)
    return rtn

def JacobiIterationErr(A, b, x0, errtol, verbose=False):
    rtn = x0.copy()
    Diag = np.diagonal(A)
    (row, col) = np.shape(A)
    D = np.zeros(shape=(row,col))
    for i in range(row):
        D[i,i] = 1/Diag[i];
    I = np.eye(row)
    negDA = np.subtract(I, np.matmul(D,A))
    DB = np.matmul(D,b)
    count=0
    while(True):
        count+=1
        prev = rtn
        rtn = np.add(np.matmul(negDA, rtn), DB)
        con = vector2Norm(np.subtract(rtn, prev))
        if(verbose):
            print(f"prev: {prev}")
            print(f"con: {con}")

        if(con <= errtol):
            break
    return (rtn, count)

#7.7 -  - Gauss Seide lIteration
#   Parameters:
#       A - numpy matrix
#       b - numpy vector(nx1)
#       x0 - single number to start itteration at
#       n - number of itterations
#   Output:
#       x - numpy matrix (nx1)
def GaussSeidelIteration(A, b, x0, n, verbose=False):
    rtn = x0.copy()
    L = np.tril(A)
    LB = np.linalg.solve(L, b)
    for _ in range(n):
        rtn = np.add(np.subtract(rtn, np.linalg.solve(L, np.matmul(A, rtn))),LB)
    return rtn

def GaussSeidelIterationErr(A, b, x0, errtol, verbose=False):
    rtn = x0.copy()
    L = np.tril(A)
    LB = np.linalg.solve(L, b)
    count=0
    while(True):
        count+=1
        prev = rtn
        rtn = np.add(np.subtract(rtn, np.linalg.solve(L, np.matmul(A, rtn))),LB)
        if(vector2Norm(np.subtract(rtn, prev)) <= 0):
            break
    return (rtn,count)

#7.7 -  - Gauss Seide lIteration
#   Parameters:
#       A - numpy matrix
#       b - numpy vector(nx1)
#       x0 - single number to start itteration at
#       n - number of itterations
#   Output:
#       x - numpy matrix (nx1)
def SORIteration(A, b, x0, n, w, verbose=False):
    rtn = x0.copy()
    L = np.tril(A, -1)
    (row, col) = np.shape(A)
    D = np.diag(np.diag(A))
    Q = np.add((1/w)*D, L)
    QB = np.linalg.solve(Q,b)
    for _ in range(n):
        rtn = np.add(np.subtract(rtn, np.linalg.solve(Q, np.matmul(A,rtn))), QB)
    return rtn;

def SORIterationErr(A, b, x0, errtol, w, verbose=False):
    rtn = x0.copy()
    L = np.tril(A, -1)
    (row, col) = np.shape(A)
    D = np.diag(np.diag(A))
    Q = np.add((1/w)*D, L)
    QB = np.linalg.solve(Q,b)
    count=0
    while True:
        count+=1
        prev = rtn
        rtn = np.add(np.subtract(rtn, np.linalg.solve(Q, np.matmul(A,rtn))), QB)
        if(vector2Norm(np.subtract(rtn, prev)) <= errtol):
            break
    return (rtn,count);

#7.8 - Multi Dimenssional Newtons Method
#   Parameters:
#       f - Function - Lmabda/Function pointer
#       errtol - Error Tolerance
#       kinv - K inverse - inverse of gradient vector
#   Output:
#       x - numpy matrix (nx1)
def multiDimNewtons(f, errtol, x0, kinv=False, verbose=False):
    if kinv==False:
        kinv = KInv(f)
    rtn = x0.copy()
    count=0
    while True:
        count+=1
        prev = rtn
        rtn = np.subtract(prev, np.matmul(kinv, multiDimF(f,prev)))
        if(vector2Norm(np.subtract(rtn, prev)) <= errtol):
            break
    return (rtn,count);

#7.8 - f(x) - Multi Dimenssional Newtons Method Helper Function
#   Parameters:
#       f - Array of Functions - Lmabda/Function pointer
#       x - Numpy matrix - values to plug into f
#   Output:
#       x - numpy matrix (nx1) - [f(x1), f(x2), ...]
def multiDimF(f,x):
    rtn = x.copy()
    for i in range(0.,x.shape[1]):
        rtn[i] = f(x[i])
    return rtn

#7.8 - K Inverse - Multi Dimenssional Newtons Method Helper Function
#   Parameters:
#       f - Array of Functions - Lmabda/Function pointer
#           Assumed that all functions have the varriables in the same order
#               If not then program might return incorrect
#   Output:
#       Kinv - numpy matrix - Gradient Matrix
#   Returns: -1 on error
def KInv(f):
    #f = []
    #for i in symf:
    #    f.append(lambdify((x,y), w))
    dim = len(f)# If all is correctly inputted this is the dim of the gradient matrix
    if dim == 0:
        print("Given matrix is empty")
        return -1
    for i in f:
        print(i.__code__.co_argcount, " ", dim)
        if(i.__code__.co_argcount != dim):
            print("Incorrect sizes, number of variables in agive equation needs to be the same as the number fo equations")
            return -1
    rtn = []    
    for i in range(dim):
        rtn.append([])
    print(rtn)
    for i in range(0,dim):
        current = f[i]
        currentSYM = implemented_function('currentSYM', current)
        print(current.__code__.co_varnames)
        tup = current.__code__.co_varnames
        for j in range(0,dim):
            print(tup[j])
            rtn[i].append(lambdify(tup[j], sy.diff(currentSYM, tup[j])))
    print(rtn)
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
    
    x,y = sy.symbols("x,y")
    
    A = lambda x,y: x**2 + y**3
    B = lambda z,a: z**3 + 3*a 
    

    
    F = implemented_function('F', A)
    
    w = sy.diff(x**2 + y**3, x)
    
    lam_f = lambdify((x,y), w)
    
    print(f"w: {w}")  
    print(f"diff: {w.subs([(x,3),(y,4)])}")
    

    f = [A, B]
    c = ch7.KInv(f)
    print(c[0][1])
