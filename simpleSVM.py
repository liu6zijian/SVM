# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:01:43 2018

@author: Lzj
"""
#import random
import numpy as np
from sklearn import svm
import cvxopt
import matplotlib.pyplot as plt

def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])



t1 = 5 + 4 * np.random.randn(2,40)
t2 = 20 + 4 * np.random.randn(2,40)

data = np.hstack((t1,t2))
plt.plot(t1[0,:],t1[1,:],'ro')
plt.plot(t2[0,:],t2[1,:],'bx')
label = np.vstack((np.ones(shape=(40,1)),-np.ones(shape=(40,1))))
K, N = data.shape
u0 = np.random.random(size = (K+1,1))
A = - np.tile(label,(1,K+1)) *np.hstack((np.ones(shape=(N,1)),data.T))
b = np.zeros(shape=(N,1))
H = np.eye(K)
H = np.vstack((np.zeros(shape = (1,K)), H))
H = np.hstack((np.zeros(shape = (K+1,1)), H))
p = np.zeros(shape=(K+1,1))
sol = quadprog(H,p,L=A,k=b)
xx = [-sol[0]/sol[1],0]
yy = [0,-sol[0]/sol[2]]
plt.plot(xx,yy,'g-.')
#plt.savefig('SVM.png',dpi=600) # save and show
plt.show()
