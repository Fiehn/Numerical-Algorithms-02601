import numpy as np
from Functions import *

def Newtonsys(FdF, X0, kmax):
    # Initialization
    X = X0
    Fx, dFx = FdF(X)
    H = np.linalg.solve(dFx,Fx) # This is the first step.
    
    # Create the array to store the iterations.
    
    Xiterations = []
    # Now iterate.
    for k in range(1, kmax+1):
        X = X - H.flatten()
        Xiterations.append(X)
        Fx, dFx = FdF(X)
        H = np.linalg.solve(dFx,Fx)
    

    return np.array(Xiterations)

def FdF(X):
    F = np.array([X[0]**2 + X[1]**2 -1,(X[0]-1/2)**2+X[1]**2 -1])

    dFx = np.array([[2*X[0],2*X[1]],[2*X[0]-1,2*X[1]]])
    return F, dFx

FdF([1,1])

Newtonsys(FdF, [0.3,1],1)


A = np.array([[1,3,5],[2,4,6],[0,5,7]])

b1 = np.array([0,1,-1])
c1 = np.array([0.1,1,-1])
c2 = np.array([0,0.999,-1.001])
c3 = np.array([0,0.998,-1.005])
c4 = np.array([0,0.899,-0.995])

x = np.linalg.solve(A,b1)

x1 = np.linalg.solve(A,c1)
x2 = np.linalg.solve(A,c2)
x3 = np.linalg.solve(A,c3)
x4 = np.linalg.solve(A,c4)

np.linalg.norm(c1-b1)/np.linalg.norm(b1)*np.linalg.cond(A)
np.linalg.norm(c2-b1)/np.linalg.norm(b1)*np.linalg.cond(A)
np.linalg.norm(c3-b1)/np.linalg.norm(b1)*np.linalg.cond(A)
np.linalg.norm(c4-b1)/np.linalg.norm(b1)*np.linalg.cond(A)

np.linalg.norm((x-x1))/np.linalg.norm(x)
np.linalg.norm(x-x2)/np.linalg.norm(x)
np.linalg.norm(x-x3)/np.linalg.norm(x)
np.linalg.norm(x-x4)/np.linalg.norm(x)


# Jacoby of the system
f1 = lambda x,y: x*np.sin(y)+x**2
f2 = lambda x,y: y*np.cos(x)-x*y-y

# Jacoby of the system in matrix form
J = lambda x,y: np.array([[np.sin(y)+2*x,x*np.cos(y)],[y*np.sin(x)-y,-np.cos(x)-x-1]])
J(np.pi/2,-np.pi/2)

np.array([[np.pi-1,0],[np.pi,-np.pi/2-1]])