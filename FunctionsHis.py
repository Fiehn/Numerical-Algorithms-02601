""" This file contains all of his functions."""

def MyHorner(a, x):
    m = len(a)
    p = a[m - 1]
    for i in range(m - 2, -1, -1):
        p = a[i] + p * x
    return p

## Root Finding
def Bisection(f, a, b, nmax):
    # Check input.
    if nmax < 1:
        raise ValueError("nmax must be a positive number")

    # Prepare to iterate.
    if a >= b:
        raise ValueError("a must be less than b")
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    X = [0] * (nmax + 1)  # Create X to store the iterations.

    # Iterate.
    for n in range(nmax):
        c = (a + b) / 2  # The midpoint.
        fc = f(c)  # The corresponding function value.
        X[n] = c
        if fa * fc < 0:
            b = c
            # fb = fc  # Note that fb = f(b) is actually not used.
        else:
            a = c
            fa = fc

    # Finish by computing the midpoint of the last interval.
    c = (a + b) / 2
    X[nmax] = c

    return X

def Newton(f, df,x0,nmax):
    X = []
    x = x0
    X.append(x)
    for i in range(nmax):
        fx = f(x)
        fp = df(x)
        x = x - fx/fp
        X.append(x)
    return X

def Secant(f, x0, x1, nmax):
    """
    This function implements the secant method to find a root of the function f.
    
    f: the function whose root we want to find
    x0: the initial guess for the root
    x1: the second initial guess for the root
    nmax: the maximum number of iterations
    
    returns: a root of the function f, or None if no root is found
    """
    # Set a tolerance for the error
    tol = 1e-6
    X = [x1]
    # Iterate nmax times
    for n in range(nmax):
        # Compute the value of the function at x0 and x1
        f0 = f(x0)
        f1 = f(x1)
        
        # Compute the secant of the function at x0 and x1
        sec = (x1 - x0) / (f1 - f0)
        
        # Compute the next guess for the root
        x2 = x1 - sec * f1
        X.append(x2)
        # Check if the error is within the tolerance
        if abs(x2 - x1) < tol:
            return X
        
        # Update x0 and x1 for the next iteration
        x0 = x1
        x1 = x2
    
    # If no root was found, return None
    return X

## Interpolation
import numpy as np
def cardinalpoly(knots, i, t):
    nodes = np.array(knots)
    t = np.array(t)
    x_i = nodes[i]
    p = 1
    for x_j in np.delete(nodes, i, axis=0):
        p *= (t - x_j) / (x_i - x_j)
    return p
def LagrangeFormInterpolation(knots, ydata, t):
    """
    LagrangeFormInterpolation: Calculates the values of the interpolating polynomial in Lagrange form

    Args:
        knots (list): [x0 x1 ... xn]   is a row of n+1 knot-values
        ydata (list): [y0 y1 ... yn]   is a row of the corresponding n+1 y-values
        t (list): [t1 ... tm]          is a row of all the m values that the inter-polating polynomial is to be evaluated in

    Returns:
        list: [P(t1) ... P(tm)]  a row with the m function values of the interpolating polynomial
    """
    
    P_val = []
    cardinals = []
    for idx in range(len(ydata)):
        cardinals.append(cardinalpoly(knots,idx,t))
    np.asarray(cardinals)
    
    P_val = np.zeros(len(t))
    for idx,y in enumerate(ydata):
        P_val = np.add(P_val,np.multiply(cardinals[idx],y))
    
    return P_val

def MinInversKvadInterpolation(f, x, n):
    # udfører n iterationer af invers
    # kvadratisk interpolation for
    # f function handtreel
    # x=[x0, x1, x2] er de initiale x-værdier
    # n er antallet af iterationer
    # MinInverKvadInterpolation returnerer en vektor X med alle n estimater
    # af roden

    fX = np.zeros(n+3)
    X = np.zeros(n+3)
    X[0:3] = x
    for i in range(3):
        fX[i] = f(x[i])
    for i in range(n):
        X[i+3] = LagrangeFormInterpolation(fX[i:i+3], X[i:i+3], [0])
        fX[i+3] = f(X[i+3])
    return X[3:], fX[3:]

def MinInversOrderKInterpolation(f, x, n):
    # MinInverKvadInterpolation udfører n iterationer af invers
    # kvadratisk interpolation for
    # f function handel
    # x=[x1,..,xk] er de k initiale x-værdier
    # n er antallet af iterationer
    # MinInverKvadInterpolation returnerer en vektor X med alle n estimater
    # af roden

    k = len(x)
    fX = np.zeros(n+k)
    X = np.zeros(n+k)
    X[0:k] = x
    for i in range(k):
        fX[i] = f(x[i])
    for i in range(n):
        X[i+k] = LagrangeFormInterpolation(fX[i:i+k], X[i:i+k], [0])
        fX[i+k] = f(X[i+k])
    return X[k:], fX[k:]

def MyTrapez(f,a,b,n):
    h = (b-a)/n
    sum_x= (1/2)*(f(a)+f(b))
    for i in range(1,n,1):
        x = a+i*h
        sum_x += f(x)
    sum_x = sum_x*h
    return sum_x

## Integrate ODE
def MyEuler(dxdt, tspan, x0, n):
    """
    Uses Euler's method to integrate an ODE
    
    Input:
    
      dxdt = function handle to the function returning the rhs of the ODE
      
      tspan = [a, b] where a and b are initial and final values of the independent variable
      
      x0 = initial value of dependent variable
     
      n = number of steps
      
    Output:
      t = vector of independent variable
     
      x = [x_0 x_1 ... x_n] vector of solution for dependent variable
    """
    a, b = tspan
    t = np.linspace(a, b, n+1) # Dette er bedre end at fremskrive t
    h = (b - a) / n # h is calculated once only
    x = np.zeros(n+1) # preallokere x det er hurtigere
    x[0] = x0 # lægger x0 som første x-værdi
    for i in range(n): # implementerer Euler's method
        x[i+1] = x[i] + dxdt(t[i], x[i]) * h
    return t, x

def MyEulerSystem(dxdt, tspan, x0, n):
    
    dim1 = x0.shape[0]
    dim2 = dxdt(tspan[0], x0).shape[0]  # a bit wasteful, but can be saved
    if dim1 - dim2 != 0:
        raise ValueError('The dimensions of x0 and the right side do not match')

    a, b = tspan
    
    t = np.linspace(a, b, n+1)
    h = (b - a) / n  # h is calculated once only
    x = np.zeros((dim1, n+1))  # preallocate x to improve efficiency
    x[:,0] = x0
    for i in range(n):  # Euler's method
        x[:,i+1] = x[:,i] + dxdt(t[i], x[:,i]) * h
    return t, x

def MyHeun(dxdt, tspan, x0, n):
    # Uses Heun's method to integrate an ODE
    # Input:
    #   dydt = function handle to the rhs. of the ODE
    #   tspan = [a, b] where a and b = initial and final values of independent variable
    #   x0 = initial value of dependent variable
    #   n = number of stemp
    # Output:
    #   t = vector of independent variable
    #   x = vector of solution for dependent variable

    a, b = tspan
    t = np.linspace(a, b, n+1)
    h = (b - a) / n
    hhalve = h / 2.0 # beregner kun h/2 én gang
    x = np.zeros(n+1) # preallokere x det er hurtigere
    x[0] = x0 # lægger x0 som første x-værdi
    for i in range(n): # Heun's method
        K1 = dxdt(t[i], x[i])
        K2 = dxdt(t[i+1], x[i] + h * K1)
        x[i+1] = x[i] + (K2 + K1) * hhalve
    return t, x

def MyRK4(dxdt, tspan, x0, n):
    a = tspan[0]
    b = tspan[1]
    t = np.linspace(a, b, n + 1)
    h = (b - a) / n
    hhalve = h / 2.0
    hbysix = h / 6.0
    x = np.ones((n + 1)) * x0 # preallocate x to improve efficiency
    for i in range(n): # implement the method
        K1 = dxdt(t[i], x[i])
        K2 = dxdt(t[i] + hhalve, x[i] + hhalve * K1)
        K3 = dxdt(t[i] + hhalve, x[i] + hhalve * K2)
        K4 = dxdt(t[i + 1], x[i] + h * K3) # avoid t(i) + h
        x[i + 1] = x[i] + hbysix * (K1 + 2 * (K2 + K3) + K4)
    
    return t, x

def MyRK4System(odefunctions,tspan,X0,n):
    a,b = tspan
    t = a
    X = X0
    h = (b-a)/n
    t_list = [t]
    for j in range(1,n+1):
        K1 = h*odefunctions(t,X)
        K2 = h*odefunctions(t+1/2*h, X+1/2*K1)
        K3 = h*odefunctions(t+1/2*h, X+1/2*K2)
        K4 = h*odefunctions(t+h,X+K3)
        X = X + 1/6*(K1+2*K2+2*K3+K4)
        t = a + j*h
        t_list.append(t)
        X0 = np.vstack((X0,X))
    return t_list, X0

def MyTaylorOrder2(f, df, tspan, x0, n):
    ti, tf = tspan
    t = np.linspace(ti, tf, n+1)
    x = np.zeros(n+1)   # preallocate x for speed
    x[0] = x0
    h = (tf - ti) / n
    for i in range(n):
        dx = f(t[i], x[i])
        ddx = df(t[i], x[i], dx)
        x[i+1] = x[i] + h * (dx + h/2 * (ddx))
    return t, x

def odesystemkanon( _ , indata ):
    """
     simultaneous second order differentials for projectile
     motion with air resistance
     output vector z has the four differential outputs
     assumed units: metres, seconds, Newtons, kg, radians
     da t som skal være første variabel ikke bruges kan 
     den erstattes af ~ i funktionsdeklereringen
    """

    g=9.81 # m/s^2
    m=550/1000 # mass of projectile, kg
    d=0.07 # diameter of spherical projectile, meters
    Cd=0.5 # assumed
    rho=1.2041 # density of air, kg/m^3
    A=(np.pi*d**2)/4 # silhouette area, m^2
    C=Cd*A*rho/2/m # the drag force constant
    z = np.zeros(4) # initialize space

    z[0] = indata[1]
    z[1] = (-C) * np.sqrt(indata[1]**2 + indata[3]**2)*indata[1]
    z[2] = indata[3]
    z[3] = (-g) + ((-C) * np.sqrt(indata[1]**2 + indata[3]**2) * indata[3])
    
    return z

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
