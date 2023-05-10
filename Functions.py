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

### Least Squares
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

### Print matrix nice
def print_table(A):
    """ Print matrix in nice format
    A: matrix
    """
    if type(A) != np.ndarray:
        A = np.array(A)
    m,n = A.shape
    for i in range(m):
        for j in range(n-1):
            print("{:10.4f}".format(A[i,j]),end=" ")
        print("{:10.4f}".format(A[i,-1]),end=" ")
        print()

def print_normal_equation(F,x,y,digits=4):
    """ Print the normal equation for a given function F
    F: function to be fitted
    x: x-values
    """
    A = np.array(list(map(F,x)))
    print_table(A.T@A)
    print(np.round(A.T@y,4))


### Print matrix to latex
def print_table_latex(A):
    """ Print normal equations in latex format
    A: matrix
    """
    if type(A) != np.ndarray:
        A = np.array(A)
    m,n = A.shape
    print(r"\begin{equation*}")
    print(r"\begin{array}{|"+n*"r"+"|}")
    for i in range(m):
        for j in range(n-1):
            print(A[i,j],end=" & ")
        print(A[i,-1],end="\\\\")
        print()
    print(r"\end{array}")
    print(r"\end{equation*}")

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

def Newtonsys(FdF, X0, kmax):
    """ Newton's method for solving systems of equations
    FdF: function returning F(x) and dF(x)
    X0: initial guess
    kmax: maximum number of iterations
    """
    X = X0
    
    Xiterations = []
    for k in range(kmax):
        Fx, dFx = FdF(X)
        H = np.linalg.solve(dFx, Fx)
        X = X - H.flatten()
        Xiterations.append(X)
    return np.array(Xiterations)

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
    return 1/(4*(n+1)) * M *h**(n+1)

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

# Trapezoidal rule for unknown function (table) Would be the composite
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

#####
# ODE 1st order
#####
# Taylor polynomial of order 2
def TaylorOrder2(f, df, x0, a, b, n):
    """ Calculates the Taylor polynomial of order 2 for f on [a,b] using n steps
        f: function as lambda function
        df: derivative of f as lambda function
        x0: point of expansion
        a: lower bound
        b: upper bound
        n: number of steps
    """
    h = (b-a)/n
    t = np.linspace(a,b,n+1) 
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        dx = f(t[i], x[i])
        ddx = df(t[i], x[i], dx)
        x[i+1] = x[i] + h * (dx + h/2 * (ddx))
    return t, x

# Taylor polynomial of order 4
def TaylorOrder4(f, df, ddf, x0, a, b, n):
    """ Calculates the Taylor polynomial of order 4 for f on [a,b] using n steps
        f: function as lambda function
        df: derivative of f as lambda function
        ddf: second derivative of f as lambda function
        x0: point of expansion
        a: lower bound
        b: upper bound
        n: number of steps
    """
    h = (b-a)/n
    t = np.linspace(a,b,n+1) 
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        dx = f(t[i], x[i])
        ddx = df(t[i], x[i], dx)
        dddx = ddf(t[i], x[i], dx, ddx)
        x[i+1] = x[i] + h * (dx + h/2 * (ddx + h/6 * dddx))
    return t, x

# euler method
def EulerODE(dxdt, x0, tspan, n):
    """ Calculates the solution of the ODE using the Euler method
        dxdt: rhs of the ODE as lambda function
        x0: point of expansion
        tspan: [a,b]
        n: number of steps
    """
    a, b = tspan
    h = (b-a)/n
    t = np.linspace(a,b,n+1) 
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        x[i+1] = x[i] + h * dxdt(t[i], x[i])
    return t, x

# Runge-kutta method of order 2
def RungeKutta2(dxdt, x0, tspan, n):
    """ Calculates the solution of the ODE using the Runge-Kutta method
        dxdt: rhs of the ODE as lambda function
        x0: point of expansion
        tspan: [a,b]
        n: number of steps
    """
    a, b = tspan
    h = (b-a)/n
    t = np.linspace(a,b,n+1) 
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        k1 = dxdt(t[i], x[i])
        k2 = dxdt(t[i] + h, x[i] + k1)
        x[i+1] = x[i] + h/2 * (k1 + k2)
    return t, x


# Runge-kutta method of order 4
def RungeKutta4(dxdt, x0, tspan, n):
    """ Calculates the solution of the ODE using the Runge-Kutta method
        dxdt: rhs of the ODE as lambda function
        x0: point of expansion
        tspan: [a,b]
        n: number of steps
    """
    a,b = tspan
    h = (b - a) / n
    h_half = h / 2.0 # for convenience
    h_sixth = h / 6.0 # for convenience
    t = np.linspace(a, b, n + 1) 
    x = np.zeros(n + 1)
    x[0] = x0
    for i in range(n):
        K1 = dxdt(t[i], x[i])
        K2 = dxdt(t[i] + h_half, x[i] + K1*h_half)
        K3 = dxdt(t[i] + h_half, x[i] + K2*h_half)
        K4 = dxdt(t[i + 1], x[i] + h*K3)
        x[i+1] = x[i] + (K1 + 2 * (K2 + K3) + K4) * h_sixth
    return t, x

# Heun method
def HeunODE(dxdt, x0, tspan, n):
    """ Calculates the solution of the ODE using the Heun method
        dxdt: rhs of the ODE as lambda function
        x0: point of expansion
        tspan: [a,b]
        n: number of steps
    """
    a, b = tspan
    h = (b-a)/n
    h_half = h/2 # for convenience
    t = np.linspace(a,b,n+1)
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n): # Heun's method
        K1 = dxdt(t[i], x[i])
        K2 = dxdt(t[i+1], x[i] + h * K1)
        x[i+1] = x[i] + (K2 + K1) * h_half
    return t, x

#####
# System of ODE'se
#####
# Convert 2 ODE's to system of ODE's
def f2(x, y):
    """ Takes two ODE's and returns the system of ODE's
        x: ODE as lambda function
        y: ODE as lambda function
    """
    def f(t, X):
        return np.array([x(t, X[0], X[1]), y(t, X[0], X[1])])
    return f

# Runge-kutta method of order 4 for system of ODE's
def RungeKutta4System(f, tspan, xi, nsteps):
    """ Calculates the solution to the system of ODE's using the Runge-Kutta method
        f: system of ODE's (use f2)
        tspan: [a,b]
        xi: array of initial conditions
        nsteps: number of steps
    """
    a,b = tspan
    h = (b-a)/nsteps
    h_half = h/2 # for convenience
    h_sixth = h/6 # for convenience
    t = np.linspace(a,b,nsteps+1)
    X = np.zeros((nsteps+1, len(xi)))
    X[0] = xi
    for i in range(nsteps):
        k1 = f(t[i], X[i])
        k2 = f(t[i] + h_half, X[i] + h_half * k1)
        k3 = f(t[i] + h_half, X[i] + h_half * k2)
        k4 = f(t[i] + h, X[i] + h * k3)
        X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4) * h_sixth

    return t, X

#########
# Error
#########
def errorConvergece(n, error): 
    """ Calculates the convergence rate of a method
    n: list of iterations
    error: list of errors
    returns: rate of convergence and order of convergence
    """
    conv = []
    r = []
    for i in range(len(n)-1):
        conv.append(np.log(error[i]/error[i+1])/np.log(n[i]/n[i+1]))
        r.append(error[i+1]/error[i])
    return conv, r

def errorConvergenceValues(x,x_true):
    """ Calcualates the convergence rate and order of a values
    x: list of approximations
    x_true: true value
    returns: rate of convergence and order of convergence
    """
    conv = []
    r = []
    errors = []
    for i in range(len(x)):
        errors.append(np.abs(x[i]-x_true))
    for i in range(len(x)-1):
        conv.append(np.log(errors[i]/errors[i+1])/np.log(x[i]/x[i+1]))
        r.append(errors[i+1]/errors[i])
    return conv, r
 
#######
# Factorization
#######

### Check SPD
def isSPD(A):
    """ Check if matrix is symmetric positive definite
    A: matrix
    """
    if type(A) != np.ndarray:
        A = np.array(A)
    if np.all(A == A.T):
        try:
            np.linalg.cholesky(A)
            print("Symmetric positive definite")
            return True
        except np.linalg.LinAlgError:
            print("Symmetric but not positive definite")
            return False
    else:
        print("Not symmetric")
        return False
    
### Cholesky factorization
def Cholesky(A):
    return np.linalg.cholesky(A)

def Cholesky2L(A):
    """ Gives back each step to finding L
    A: matrix
    """
    if type(A) != np.ndarray:
        A = np.array(A)
    n = len(A)
    L = np.zeros((n,n))
    L_sym = [["" for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i+1):
            if i == j:
                L[i,j] = np.sqrt(A[i,j] - np.sum(L[i,:j]**2))
                if np.sum(L[i,:j]**2)==0:
                    L_sym[i][j] = f"sqrt({A[i,j]})"
                else: 
                    L_sym[i][j] = f"sqrt({A[i,j]} - {np.sum(L[i,:j])}^2)"
            else:
                L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j]))/L[j,j]
                if np.sum(L[i,:j]*L[j,:j])==0:
                    L_sym[i][j] = f"{A[i,j]}/{L[j,j]}"
                else:
                    L_sym[i][j] = f"({A[i,j]} - {np.sum(L[i,:j]*L[j,:j])})/{L[j,j]}"
    return L, L_sym

### LU factorization
def LU(A):
    return np.linalg.lu(A)

