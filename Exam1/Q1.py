"""
@author: grant
7.5. 12 & 13
"""
import math
import numpy as np
import matplotlib.pyplot as plt

"""
The code is split into to parts, it made my life easier. YOu can see the next wave of graphs by pressing enter in the 
command line and it will go on to the next algorithm to run.

After running my pogram for Sn and other matrix functions I noticed a serious trend between the estimation and real conditionj number 
But if you look at the y axis you can see that the estimation is off by quiet a bit, but the Sn 
kept the same curve of increasing just not the same scaling. Hn and Kn both expereinced a good amount of error aswell but did not keep
the same shape as their non-estimation verrssions. An and Tn but were relitively close to their non-estimation verssions but did not
maintain the same curve. Comparing the graphs from Sn and Hn both interests me and makes sense. It make sense because they are similar 
down the 3 center diagnals so it makes sense that they both see high errors. But it itnerests me because Sn keeps it's curve constant from
the estimation to non-estimations and only differs by a scalor. Also Hn has these large spikes of  both in the estimation and not. But they
do not share n's for the spikes while Sn does not spike or get even close to the same k values.


"""



def HnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            rtn[i,j] = 1.0/( i+j+1 )
    return rtn

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

def SnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i==j:
                rtn[i-1,j-1] = 1.0/(i+j-1)
            elif abs(i-j)==1:
                rtn[i-1,j-1] = 1.0/(i+j-1)
    return rtn
                
def vectorInfinityNorm(v):
    return np.max(np.abs(v))   
def matrixInfinityNorm(A_):
    A = np.copy(np.abs(A_))
    rowsum = np.sum(A,axis=1).tolist()
    return np.max(rowsum)


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


def conditionNumber(A, Norm):
    Ainv = np.linalg.inv(A)
    return Norm(A) * Norm(Ainv)

def conditionNumberEsitimation(A):
    (row, col) = np.shape(A)
    alpha = matrixInfinityNorm(A)
    
    (X,idx) = LU(A)
    
    y = np.random.rand(row,1)#This can be cahnge TODO
    for i in range(5):
        y = y/vectorInfinityNorm(y)
        y = LUsolver(X,y,idx)
    
    return alpha * vectorInfinityNorm(y)




matrix = [TnMatrix, KnMatrix, HnMatrix, AnMatrix, SnMatrix]
matrixName = ["Tn", "Kn", "Hn", "An", "Sn"]
# The condition number (not esimate edition)
"""
for i in range(5):
    print(matrixName[i],":")
    print(matrix[i](4))
    print()
print()

input('Print part 1 (Condition Number no estimation): ')
"""
for i in range(5):
    K=[]
    for j in range(4,21):
        K.append(conditionNumber( matrix[i](j), vectorInfinityNorm ))
    plt.plot(np.linspace(4, 20, 17), K)    
    plt.suptitle(matrixName[i]+" Cond num no est")
    plt.xlabel("n")
    plt.ylabel("K")
    plt.show()


input('Print part 2 (Condition Number estimation): ')

# The condition number (estimated)
avgCount=10
for i in range(5):
    K=[]
    for j in range(4,21):
        sum=0
        for avg in range(avgCount):   
            sum += conditionNumberEsitimation( matrix[i](j) )
        K.append(sum/avgCount)
    plt.plot(np.linspace(4, 20, 17), K)
    plt.suptitle(matrixName[i]+" Cond num est")
    plt.xlabel("n")
    plt.ylabel("K")
    plt.show()         
    


    
    

