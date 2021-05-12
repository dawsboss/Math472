#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:17:18 2021

@author: grant
"""

import numpy as np
import ch8.Library.ch8LIB as ch8
import ch7.Library.ch7LIB as ch7
import matplotlib.pyplot as plt

A = np.matrix([[4, 1, 0],
               [1, 4, 1],
               [0, 1, 4]])
eig,eigvect = np.linalg.eig(A)
largEig,blah1,_ = ch8.BasicPowerMethod(A,10**-10)
print(f"Basic Power Method: {largEig}")
print(f"numpy biggest: {eig[np.argmax(abs(eig))]}\n")

eigs=[]
itter=[]

for i in range(2,21):
    A = ch7.HnMatrix(i)
    eig,eigvect = np.linalg.eig(A)
    print(f"Made A size: {i}")
    largEig,blah1,_ = ch8.BasicPowerMethod(A,10**-10)
    print("Made bigeig")
    smolEig,blah2,_ = ch8.InversePowerMethod(A,10**-10)
    eigs.append(smolEig)
    itter.append(i)
    if(i<5):
        print(A,"\n")
    print(f"Inverse Power Method: {smolEig}")
    print(f"numpy biggest: {eig[np.argmax(abs(eig))]}\n")
    print(f"Baseic Power Method: {largEig}")
    print(f"numpy smallest: {eig[np.argmin(abs(eig))]}\n")
    print(f"\n")
    

plt.title("Inverse Power Method")
plt.plot(itter, eigs)
    
    

