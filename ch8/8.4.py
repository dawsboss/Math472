#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:20:06 2021

@author: grant
"""
import numpy as np
import ch8.Library.ch8LIB as ch8
import ch7.Library.ch7LIB as ch7

np.set_printoptions(formatter={'float':'{:0.3f}'.format})

A = np.matrix([[10, -2, 3, 2, 0],
               [-2, 10, -3, 4, 5],
               [3, -3, 6, 3, 3],
               [2, 4, 3, 6, 6],
               [0, 5, 3, 6, 13]], dtype=float)

Q, R = ch8.get_QR(A)

A_=R@Q

for i in range(30):
    Q_, R_ = ch8.get_QR(A_)
    A_ = R_@Q_

print(ch8.get_eigvals(A))


print(A_)
print(np.diag(A_))