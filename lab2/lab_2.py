
# coding: utf-8

# In[1]:

import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D


### FUNCTION INITIALISATION
def f_var6(x):
    return x[0]**2 + 4.0 * x[1]**2 + 0.001 * x[0] * x[1] - x[1]    
def f_test(x):
	 return x[0]+x[1]

f_type = 'quadratic'
if (f_type == 'quadratic'):
    A = np.array([[2, 0.001], [0.001, 8]], dtype = float)
    b = np.array([0, -1], dtype = float)

### INITIAL POINT
x0 = np.array([10,10], dtype = float)

### SETTINGS
MAX_COUNT_ITER = 10000
EPS = 10**(-5)
h = np.array([0.01,0.01], dtype = float)


# In[2]:
#3D Visualization
def makeData ():
    x = np.arange (-50, 50, 0.5)
    y = np.arange (-50, 50, 0.5)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = f([xgrid, ygrid])
    return xgrid, ygrid, zgrid

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)
axes.plot_surface(x, y, z)
pylab.show()


# In[ ]:
### APPROXIMATE DERIVATIVES
def df1(x,h1):
    return (f([x[0] + h1, x[1]]) - f([x[0] - h1, x[1]])) / (2 * h1)
def df2(x,h2):
    return (f([x[0], x[1] + h2]) - f([x[0], x[1] - h2])) / (2 * h2)

def df11(x,h1):
	 return (f([x[0] + h1, x[1]]) - 2 * f(x[0],x[1])+ f([x[0] - h1, x[1]])) / (h1 * h1)
def df22(x,h2):
	 return (f([x[0], x[1] + h2]) - 2 * f(x[0],x[1])+ f([x[0], x[1] - h2])) / (h1 * h1)
def df12(x,h1,h2):
	 return (f([x[0] + h1, x[1] + h2]) + f([x[0] - h1, x[1] - h2])
	 		 - f([x[0] + h1, x[1] - h2]) - f([x[0] - h1, x[1] + h2])) / (2 * h1 * h1)
def grad(x,h):
    return np.array([df1(x,h[0]), df2(x,h[1])])
def gesse(x,h):
    return np.array([ [df11(x,h[0]), df12(x,h[0],h[1])], [df12(x,h[0],h[1]), df22(x,h[1])] ] )

### TRUE GRADIENT (ONLY FOR QUADRATIC FUNCTIONS)
def truegrad(A,x):           
    return A.dot(x)+b

### STOP CONDITIONS
def stop1(x1,x2,k):
    plt.ylabel('|| x_new - x_old || ')
    d = norm(x2-x1, ord=2)
    plt.scatter(k, d)
    return d<=EPS

def stop2(x1,x2,k):
    plt.ylabel('| f(x_new) - f(x_old) | ')
    d = abs(f(x2)-f(x1))
    plt.scatter(k, d)
    return d<=EPS

def stop3(x,h,k):
    plt.ylabel('|| f\'(x_new) || ')
    d = norm(df(x,h), ord=2)
    plt.scatter(k, d)
    return d<=EPS

### THE NEWTON METHOD
def newton_method(x0,h):
    fout = open('output.txt', 'w')
    fout.write('The initial point is ({x}, {y})\n\n'.format(x=x0[0], y=x0[1]))
    x_new = x0
    k = 0
    plt.xlabel('iteration')
    while (k<MAX_COUNT_ITER):
        x_old = x_new
        matr = inv(gesse(x_old,h))
        step = - matr.dot(gradf(x_old,h))
        x_new = x_old + step
        k += 1
        fout.write('{iter:>3}. alpha = {al:<17.15f},   x_{iter:<3} = ({x:>18.15f}, {y:>18.15f})\n'.format(iter=k, x=x_new[0], y=x_new[1], al=alpha))    
        if (stop1(x_old,x_new,k)):     ### STOP CONDITION 1
        #if (stop2(x_old,x_new,k)):    ### STOP CONDITION 2
        #if (stop3(x_new,h,k)):        ### STOP CONDITION 3
            break
    print('Gradient method found approximate solution in {} iterations'.format(k))
    fout.write('\nThe approximate solution of the problem is ({x:>10.7f}, {y:>10.7f})\n'.format(x=x_new[0], y=x_new[1]))
    fout.write('The value of function in this point is {v:>10.7f}\n'.format(v=f(x_new)))
    fout.close()
    return x_new


 
### PROGRAM    
f = f_var6
minim = gradient_method(x0,h)
print('The initial point is ({x}, {y})'.format(x=x0[0], y=x0[1]))
print('The approximate solution of the problem is ({x:>10.7f}, {y:>10.7f})'.format(x=minim[0], y=minim[1]))
print('The value of function in this point is {v:>10.7f}'.format(v=f(minim)))
plt.show()

#trueminim = np.array([-1000.0/15999999, 2000000.0/15999999])
#print('The true solution of the problem is ({x:>10.7f}, {y:>10.7f})'.format(x=trueminim[0], y=trueminim[1]))
#print('The value of function in this point is {v:>10.7f}'.format(v=f(trueminim)))
print()

