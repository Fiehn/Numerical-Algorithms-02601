import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Functions import *

dxdt = lambda t,x: t*x
x0 = 1
x_true = lambda t: np.exp(t**2/2)
x2 = x_true(2)

eu = EulerODE(dxdt, x0, [0, 2], 40)
eu2 = eu[1][np.where(eu[0] == 2)] # Find x(2) for the Euler method

he = HeunODE(dxdt, x0, [0, 2], 20)
he2 = he[1][np.where(he[0] == 2)] # Find x(2) for the Heun method

f = lambda t,x: t*x
df = lambda t,x,dx: x + dx
ta = TaylorOrder2(f, df, x0, 0, 2, 20)
ta2 = ta[1][np.where(ta[0] == 2)] # Find x(2) for the Taylor method


rk = RungeKutta4(dxdt, x0,[0,2], 10)
rk2 = rk[1][np.where(rk[0] == 2)] # Find x(2) for the Runge-Kutta method

# Find the error for each method
eu_error = abs(x2 - eu2)
he_error = abs(x2 - he2)
ta_error = abs(x2 - ta2)
rk_error = abs(x2 - rk2)
# Print the error for each method into a latex table for the report
print('Euler & %.4f \\\\' % (eu_error))
print('Heun & %.4f \\\\' % (he_error))
print('Taylor & %.4f \\\\' % (ta_error))
print('Runge-Kutta & %.4f \\\\' % (rk_error))


### 19.2 ###
# Shooting method for the boundary value problem of non linear ODEs

# ODEs


def phi(z):
    # ODEs to solve 
    f = lambda t,y: [y[1], (y[0]**2 - y[1] - t)/7] # takes [y, u] and returns [u, u']
    # Initial conditions
    xspan = [0,2]
    y0 = [7,z]
    
    sol = solve_ivp(f, xspan, y0)
    return sol.y[0][-1]


sols = []
zs = np.linspace(-60,0,61)
for i in zs:
    sols.append(phi(i))
# grid
plt.grid()
plt.plot(zs, sols)
plt.xlabel('z')
plt.ylabel('phi(z)')
plt.show()


def Bisection(f, a, b, nmax):
    """ Bisection method for finding roots of f(x)=0
    f: function in lambda form
    a: left endpoint of interval
    b: right endpoint of interval
    nmax: maximum number of iterations
    """
    if nmax < 1:
        raise ValueError("nmax must be a positive number")
    if a >= b:
        raise ValueError("a must be less than b")
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    X = [0] * (nmax + 1)  

    for n in range(nmax):
        c = (a + b) / 2  # The midpoint.
        fc = f(c)  
        X[n] = c
        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
    c = (a + b) / 2
    X[nmax] = c
    return X

f = lambda z: phi(z) - 2
root1 = Bisection(f, a = -50, b = -30, nmax = 100)[-1]
root2 = Bisection(f, a = -10, b = 0, nmax = 100)[-1]

print('Root 1: %.4f' % (root1))
print('Root 2: %.4f' % (root2))


# Plot the solution
f = lambda t,y: [y[1], (y[0]**2 - y[1] - t)/7]
# Initial conditions
xspan = [0,2]
y0 = [7,root1]
sols = []
ts = np.linspace(0,2,100)
sol = solve_ivp(f, xspan, y0,t_eval=ts)

plt.plot(ts, sol.y[0])
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()









