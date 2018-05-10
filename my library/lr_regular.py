# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:09:48 2018

@author: e0046971
"""
import numpy as np
class lr():
    ''' This is Salamat Burzhuev's library for logistic regression for binary
    classification
    X- matrix to be fitted where number of columns are number of examples
    while number of rows is features (n_x,m)
    Y- results is (1,m) matrix
    '''
    def __init__(self,X,Y,X_cv=[],Y_cv=[],CV=False,learning_rate=0.01,iterations=100,lambd=0.0001):
        self.X=X
        self.Y=Y
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.W,self.b=self.initialize()
        self.error=[]
        self.CV=CV
        self.lambd=lambd
        # decided to put in order to have get error from cross validation if X_cv and Y_cv is provided
        if self.CV : #true of false
            self.X_cv=X_cv
            self.Y_cv=Y_cv
            self.error_cv=[]
    def initialize(self):
        ''' W should same length as features X, (n_x,1), while b same number as
        number of train data, (1,m). However, b is same for all training examples
        We can use python broadcasting , and just define b is one number, which is zero
        
        '''
        n_x=self.X.shape[0]
        W=np.zeros((n_x,1)) 
        b=0
        return W,b
    def fit(self):
        W,b=self.W,self.b # getting W, b values
        for i in range(self.iterations):
            Y_hat=self.sigmoid(self.Z(self.X,W,b))
            J=self.cost(Y_hat)
            self.error.append(J)
            if self.CV: # True or False
                Y_hat_cv=self.sigmoid(self.Z(self.X_cv,W,b))
                J_cv=self.cost_cv(Y_hat_cv)
                self.error_cv.append(J_cv)
            dW,db=self.grad(Y_hat,W,b)
            W=W-self.learning_rate*dW
            b=b-self.learning_rate*db
            self.W=W
            self.b=b

    def predict_proba(self,X):
        W,b=self.W,self.b
        Y_prob=self.sigmoid(self.Z(X,W,b))
        return Y_prob
    
    def predict(self,X):
        Y_prob=self.predict_proba(X)
        Y_pred=(Y_prob>0.5).astype(int)
        return Y_pred
        
    def Z(self,X,W,b):
        Z=np.dot(W.T,X)+b
        return Z
        
    def sigmoid(self,Z): # computing sigmoid function
        return (1/(1+np.exp(-Z)))
    
    def cost(self,Y_hat):
        m=self.X.shape[1]
        J=(-1/m)*np.sum(self.Y*np.log(Y_hat)+(1-self.Y)*np.log(1-Y_hat))+np.squeeze((self.lambd/(2*m))*np.dot(self.W.T,self.W))
        # np.squeeze to reduce dimentionlity, e.i, np.squeeze(array([[1]]))=array(1)
        return J
    
    
    def cost_cv(self,Y_hat):
        m=self.X_cv.shape[1]
        J=(-1/m)*np.sum(self.Y_cv*np.log(Y_hat)+(1-self.Y_cv)*np.log(1-Y_hat))
        return J
    
    def grad(self,Y_hat,W,b):
        m=self.X.shape[1]
        ''' I will write dZ,dW,db are derivatives of cost function with
        respect to Z,W,b. a = sigmoid(z) = Y_hat
        '''
        ''' corresponding steps. I remove them to make code faster. 
        da=-self.Y/Y_hat+(1-self.Y)/(1-Y_hat)
        dZ=da*(self.sigmoid(self.Z(W,b))*(1-(self.sigmoid(self.Z(W,b)))
        # Since  a=self.sigmoid(self.Z(W,b), dZ=Y_hat-Y 
        '''
        ''' chain rule was implemented
        dJ/da=(-1/m)((y/a)-(1-y)/(1-a))
        dJ/dz=(dJ/da)(da/dz)=1/m(a-Y)
        
        *** da/dz=a(1-a) derived from sigmoid function=1/(1+exp(-z))
        dJ/dW=(dJ/dz)(dz/dW)=1/m(a-y)*X
        this should be same shape as W so
        dJ/dz is (1,m) W is n_x,m So, dJ/dW=np.dot(X,((1/m)*(a-y)).T)
        
        dJ/db=(dJ/dz)(dz/db)=(dJ/dz)
        b is same size as y (1,m)
        '''
        dZ=(1/m)*(Y_hat-self.Y)
        db=np.sum(dZ)
        dW=np.dot(self.X,dZ.T)+(self.lambd/m)*W
        return dW,db