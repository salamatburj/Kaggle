# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:09:48 2018

@author: e0046971
"""
import numpy as np
class multiclass_reg():
    ''' This is Salamat Burzhuev's library for multiclass regression using softmax.
    X- matrix to be fitted where number of columns are number of examples
    while number of rows is features (n_x,m)
    Y- results is (1,m) matrix which will be converted to Y_m which is (k,m)
    where k is number of unique features of y
    '''
    
    def __init__(self,X,Y,learning_rate=0.01,iterations=100,lambd=0.001):
        self.X=X
        self.Y,self.unique=self.convert_y(Y)
        self.lambd=lambd
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.W,self.b=self.initialize()
        self.error=[]

# this function converts Y(1,m) to Y(k,m) where k is number of unique features

    def convert_y(self,Y):
        unique=np.unique(Y)
        unique=unique.reshape((unique.shape[0],1)) #reshape to (n,1)
        Ym=np.dot(np.ones((unique.shape[0],1)),Y)
        Ym=(Ym==unique).astype(int)
        return Ym,unique
    
    def initialize(self):
        ''' W should  (n_x,k), while b same number as
        number of train data, (k,1). However, b is same for all training examples
        We can use python broadcasting , which will conver it to (k,m)
        
        '''
        k=self.Y.shape[0]
        n_x=self.X.shape[0]
        W=np.zeros((n_x,k)) 
        b=np.zeros((self.Y.shape[0],1)) # shape k,1 with broadcasting k,m
        return W,b
    
    def fit(self):
        W,b=self.W,self.b
        for i in range(self.iterations):
            Y_hat=self.softmax(self.Z(self.X,W,b))
            J=self.cost(Y_hat)
            self.error.append(J)
            dW,db=self.grad(Y_hat,W,b)
            W=W-self.learning_rate*dW
            b=b-self.learning_rate*db
            self.W=W
            self.b=b

    def predict_proba(self,X):
        W,b=self.W,self.b
        Y_prob=self.softmax(self.Z(X,W,b))
        return Y_prob
    
    def predict(self,X):
        Y_prob=self.predict_proba(X)
        Y_pred=np.dot(self.unique.T,(Y_prob==np.max(Y_prob,axis=0)))
        
        return Y_pred
        
    def Z(self,X,W,b): # same shape as Y
        Z=np.dot(W.T,X)+b
        return Z
    
    def softmax(self,Z):
        Y_hat=np.exp(Z)/np.sum(np.exp(Z),axis=0) # sum should be done for each training sample
        return Y_hat
        
    
    def cost(self,Y_hat):
        m=self.X.shape[1]
        reg_contr=(self.lambd/(2*m))*np.sum(self.W*self.W)
        J=(-1/m)*np.sum(np.multiply(self.Y,np.log(Y_hat)))+reg_contr # element wise multiplication
        return J
    
    
    def grad(self,Y_hat,W,b):
        m=self.X.shape[1]
        ''' I will write dZ,dW,db are derivatives of cost function with
        respect to Z,W,b.  Y_hat = softmax(z) 
        
        dJ/dZ= (-1/m)*(Y_hat-Y)
        dJ/dW=(dJ/dZ)*(dZ/dW)
        dJ/db=(dJ/dZ)*(dZ/db)=dJ/dz
        '''

        dZ=(1/m)*(Y_hat-self.Y)
        db=np.sum(dZ,axis=1).reshape(dZ.shape[0],1)
        dW=np.dot(self.X,dZ.T)+(self.lambd/m)*W
        return dW,db