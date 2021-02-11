#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:40:57 2021

@author: grant
3.9
"""
def fixed_point (x0, f, n, verbose=False):
    retval = x0
    
    if verbose:
        print(f"Init retval : {retval}")
    for i in range(n):
        retval = f(retval)
        if verbose:
            print(f"Iteration #{i} has value: {retval}")
    return retval