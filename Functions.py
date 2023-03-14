#####
# This file contains all the functions used in the course
import numpy as np
#####

#####
# General Purpose
#####

### Horners algorithm for polynomial evaluation
def Horners(a,x):
    """ Horner's algorithm for polynomial evaluation
    a: coefficients of polynomial
    x: point at which to evaluate
    """
    n = len(a)-1
    a.reverse()
    p = a[0]
    for i in range(n):
        p = a[i+1]+p*x
    return p

### Fitting Least Squares
# Function to be fitted:
F = lambda x: [1,x,np.sin(x)**2,np.cos(x)**2]
# Function after fitting:
Fc = lambda x,c: c[0]+c[1]*x+c[2]*np.sin(x)**2+c[3]*np.cos(x)**2
def fit(F,Fc,x,y):
    """ Fit a function to data using least squares
    F: function to be fitted
    Fc: function after fitting
    x: x-values
    y: y-values
    """
    A = np.array(list(map(F,x)))
    c = np.linalg.solve(A.T@A, A.T@y)
    return Fc(x,c),c

#####
# Root finding
#####
# Newton's method for finding roots
def Newton(f,df,x0, nmax, tol=1e-6,true_root=None):
    """
    Newton's method for finding roots of f(x)=0
    f: function in lambda form
    df: derivative of f in lambda form
    x0: initial guess
    nmax: maximum number of iterations
    tol: tolerance for stopping
    true_root: true root of f(x)=0
    """
    x = x0
    X = [x]
    for i in range(nmax):
        if true_root is not None:
            if np.abs(x-true_root) < tol:
                break
        fx = f(x)
        fp = df(x)
        x = x - fx/fp
        X.append(x)
    return X

# Bisection method for finding roots
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

# Secant method for finding roots
def Secant(f,x0,x1,nmax):
    """ Secant method for finding roots of f(x)=0
    f: function in lambda form
    x0: initial guess
    x1: initial guess
    nmax: maximum number of iterations
    """
    a = x0
    b = x1
    fa = f(a)
    fb = f(b)
    X = [a,b]
    for _ in range(nmax):
        d = fb*(b-a)/(fb-fa)
        a = b
        fa = fb
        b = b - d
        fb = f(b)
        X.append(b)
    return X 

#####
# Interpolation
#####

# Newton's interpolation using divided difference
def NewtonDD(x, y, t, n): # page 165 in E. Ward Cheney and David Kincaid, Numerical Mathematics and Computing, 7th edition
    """Newton interpolation polynomial using divided differences
    x, y: data points
    t: interpolation points
    n: number of data points
    """
    # divided difference table
    d = np.zeros((n, n))
    d[:, 0] = y
    for j in range(1, n):
        for i in range(n-j):
            d[i, j] = (d[i+1, j-1] - d[i, j-1]) / (x[i+j] - x[i])
    # evaluate polynomial
    p = d[0, 0]
    for j in range(1, n):
        pt = d[0, j]
        for k in range(j):
            pt = pt * (t - x[k])
        p = p + pt
    # coeficients
    c = d[0, :n]
    return p, c

# Cardinal polynomial for Lagrange interpolation
def CardinalPolynomial(x, i, t):
    """Lagrange cardinal polynomial
    x: list of nodes(knots)
    i: index of the node
    t: points to evaluate the polynomial as list
    """
    l = lambda k: np.prod([(k - x[j]) / (x[i] - x[j]) for j in range(len(x)) if j != i])
    return np.array([l(k) for k in t])

def InterpolerLagrangeForm(x, y, t):
    """ Calculates the values of the interpolating polynomial in Lagrange form
        x: list of nodes(knots)
        y: list of y-values
        t: points to evaluate the polynomial as list
        
    Returns:
        list: [P(t1) ... P(tm)]  a row with the m function values of the interpolating polynomial
    """
    cardinals = CardinalPolynomial(x,0,t)
    for i in range(1,len(x)):
        cardinals = np.vstack((cardinals,CardinalPolynomial(x,i,t)))

    cardinals = cardinals.T
    P_val = np.zeros(len(t))
    for idx in range(len(t)):
        P_val[idx] = np.sum(np.multiply(cardinals[idx],y))
    return P_val

# Second intepolation error sentence
def SecondInterpolationError(n,M,h):
    """ Second interpolation error sentence
    n: order of the polynomial
    M: fourth derivative of the function
    h: step size
    """
    return 1/(4*(n+1))*M*h**(n+1)

# Inverse Quadratic Interpolation
def InverseQuadraticInterpolation(f,xGuess,n):
    """ Inverse Quadratic Interpolation
    f: function
    xGuess: list of 3 initial guesses
    n: number of iterations"""
    output = []
    for _ in range(n):
        x = list(map(f,xGuess))
        x3 = InterpolerLagrangeForm(x,xGuess,[0])
        xGuess = [xGuess[1],xGuess[2],x3[0]]
        output.append(x3[0])
    return output 

# General Inverse Interpolation
def InverseInterpolation(f,xGuess,n):
    """
    Inverse Interpolation
    f: function
    xGuess: list of initial guesses
    n: number of iterations"""
    X = []
    for _ in range(n):
        x = list(map(f,xGuess))
        x_next = InterpolerLagrangeForm(x,xGuess,[0])
        xGuess = np.append(xGuess[1:],(x_next[0]))
        X.append(x_next[0])
    return X

#####
# numerical integration
#####

# Trapezoidal rule for known function
def Trapezoidal(f, a, b, n):
    """Calculates the integral of f on [a,b] using the trapez formula
        f: function
        a: lower bound
        b: upper bound
        n: number of steps
    """
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = f(x)
    return h/2*(y[0]+y[-1]+2*np.sum(y[1:-1]))

# Trapezoidal rule for unknown function (table)
def TrapezTable(y,a,b,n):
    """Calculates the integral of f on [a,b] using the trapez formula
        y: list of function values
        a: lower bound
        b: upper bound
        n: number of steps
    """
    h = (b-a)/n
    return h/2*(y[0]+y[-1]+2*np.sum(y[1:-1]))

# Simpson's rule for known function
def Simpson(f, a, b, n):
    """ Calculates the integral of f on [a,b] using the Simpson formula"""
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = f(x)
    return h/3*(y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-1:2]))

# Simpson's rule for unknown function (table)
def SimpsonTable(y,x,a,b):
    """Calculates the integral of f on [a,b] using the Simpson formula
        y: list of function values
        x: list of nodes
        a: lower bound
        b: upper bound
    """
    n = len(x)
    h = (b-a)/n
    return h/3*(y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-1:2]))







