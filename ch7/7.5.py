"""
@author: grant
7.5. 12 & 13
"""
import math
import numpy as np
import ch7.Library.ch7LIB as ch7
import matplotlib.pyplot as plt

matrix = [ch7.TnMatrix, ch7.KnMatrix, ch7.HnMatrix, ch7.AnMatrix]
matrixName = ["Tn", "Kn", "Hn", "An"]
# The condition number (not esimate edition)

for i in range(4):
    print(matrixName[i],":")
    print(matrix[i](4))
    print()
print()

input('Print part 1 (Condition Number no estimation): ')

for i in range(4):
    K=[]
    for j in range(4,21):
        K.append(ch7.conditionNumber( matrix[i](j), ch7.vectorInfinityNorm ))
    plt.plot(np.linspace(4, 20, 17), K)    
    plt.suptitle(matrixName[i])
    plt.xlabel("n")
    plt.ylabel("K")
    plt.show()


input('Print part 2 (Condition Number estimation): ')

# The condition number (estimated)
avgCount=10
for i in range(4):
    K=[]
    for j in range(4,21):
        sum=0
        for avg in range(avgCount):   
            sum += ch7.conditionNumberEsitimation( matrix[i](j) )
        K.append(sum/avgCount)
    plt.plot(np.linspace(4, 20, 17), K)
    plt.suptitle(matrixName[i])
    plt.xlabel("n")
    plt.ylabel("K")
    plt.show()         
    

input('Print part 3 (Gaussian Growth): ')    


for i in range(4):
    K=[]
    for j in range(4,21):
        K.append(ch7.growthFactor( matrix[i](j)))
    plt.plot(np.linspace(4, 20, 17), K)    
    plt.suptitle(matrixName[i])
    plt.xlabel("n")
    plt.ylabel("K")
    plt.show()
    
    

