#####
# This file contains all the functions used in the course
#####

#####
# General Purpose
#####

# Horners algorithm for polynomial evaluation
def Horners(a,x):
    n = len(a)-1
    a.reverse()
    p = a[0]
    for i in range(n):
        p = a[i+1]+p*x
    return p






