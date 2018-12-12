#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:30:01 2018

@author: nus
"""

import numpy as np

y=np.array([[1,2,3,1,2,2,2,3,4]])
unique=np.unique(y)
unique=unique.reshape((unique.shape[0],1)) #reshape to (n,1)
ym=np.dot(np.ones((unique.shape[0],1)),y)
ym=(ym==unique).astype(int)

def softmax(W,X):
    