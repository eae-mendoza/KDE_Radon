#loading in libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.special import gamma
import random
from scipy.integrate import quad,dblquad,tplquad
import scipy.integrate as integrate
import time
from sklearn import preprocessing
import sympy
import sympy as sy
from scipy.signal import *
import scipy.optimize as scop
from multiprocessing import Pool
import math
import sklearn
import random
import string
from timeit import default_timer as timer
from matplotlib.lines import Line2D

#Defining the integrand necessary in the computation of the normalization constant C
def integrand(tht_p,tht,h):
    h=h**2
    a=np.array([math.sin(tht),math.cos(tht)])
    b=np.array([math.sin(tht_p),math.cos(tht_p)])
    x=1-np.dot(b,a)
    return ((math.sqrt(2*math.pi)*h)**-1)*np.exp(-((x)**2/(2*h**2)))

#Defining the normalization constant, 'quad' is the integrating function in scipy
def C(h,d,tht):
    val=h**(d-1)*(quad(integrand,0,2*math.pi,args=(tht,h))[0])**-1
    return val

#Defining the dot-product <sample,discretezation>
def prod(sample,theta):
    a=np.zeros((len(sample),1))
    for i in range(0,len(sample)):
        a[i]=np.dot(sample[i],theta)
    return a

#Defining the kernel function
def K(sample,theta,h):
    k=norm.pdf(1-prod(sample,theta),0,h)
    return k.sum()

#Defining the function f_theta to evaluate single points
def f(theta,sample,h,tht):
    d=len(sample.T)
    val=C(h,d,tht)*len(sample)**-1*h**(-(d-1))*K(sample,theta,h)
    return val

def fhat_theta(sample,h,tht):
    result=np.array([f(i,sample,h,tht) for i in sample])
    return result

#Estimating over the whole discretization range
def f_beta_estimates(xy_sample,dist_range,d,h,C_rad,probabilities):
    f_estimates=np.array([[fbetahat(xy_sample,dist_range[i][j],d,h,C_rad,probabilities) for i in range(0,len(dist_range))] for j in range(0,len(dist_range))])
    return f_estimates

#--------------------#
#Function Defining Block
#--------------------#


#Volume of a n-dimensional sphere with radius 1
def Vn(n):
    return math.pi**(n/2)/gamma(n/2+1)

#Constant c_d^-1
def cd_inv(d):
    if d%2==0:
        return -((-1)**(d/2)*2**(-d)*math.pi**(1-d))
    else:
        return ((-1)**((d-1)/2)*2**(-d)*math.pi**(1-d))
#Defining the kernel to be used: the N(0,1)    
def std_norm(x): 
    return ((math.sqrt(2*math.pi))**-1)*sy.exp(-((x)**2/2))    
    
#Normalization constant for phi
#This is the constant Vol(S_(d-1))*integral(r^(d-1)*phi(r)dr)
#The argument it takes is d=dimension of the problem
#the [0] index takes only the first item in the value returned by quad()
#quad() returns two values, the value of the integral and the estimated numerical error
def C_radtf(d):
    def integrand_C(r):
        return r**(d-1)*std_norm(r)
    return Vn(d-1)*quad(integrand_C,0,np.inf)[0]

#This function just returns the dot product of two vectors
#After consideration, it seems that this was an unncessary step that could be removed
#But it doesn't really add anything to computation time so I will change it later
def prod2(sample,theta):
    a=np.dot(sample,theta)
    return a

def phi_tilde(s):
    return -2**-1*math.exp(-s**2/2)*s  

def constant(d,h,C):
    val=h**-(d)*C**-1*cd_inv(d)*Vn(d-2)
    return val

def hilbert_interpolator(s,srange,hilb):
    s_index=np.argmin(abs(srange-s))
    return hilb[s_index]

#function that evaluates s_tilde
def s_tilde(s,theta,t,h):
    return (s-np.dot(theta,t))/h


def fbetahat(sample,t,d,h,C_radtf,probabilities,stilde_range,hilbert_array):
    constants=constant(d,h,C_radtf)
    fthetas=probabilities
    hilb_phi=np.array([hilbert_interpolator(s_tilde(i[2],i[0:2],t,h),stilde_range,hilbert_array) for i in sample])
    test=constants*hilb_phi/fthetas
    return np.maximum(0,sum(test)/len(sample))


#Defining the discretezation points (n x n x n) matrix
def dt_mtx(a,b,discretezation):
    x1=np.arange(a,b,discretezation)
    x2=np.arange(a,b,discretezation)
    coord_mtx=np.zeros((len(x1),len(x2),2))
    for i in range(0,len(x1)):
        coord_mtx[:,i]=np.array((x1,np.full(len(x1),x2[i]))).T
    return coord_mtx


def ransample_bivar(n,pi,mu,sigma):
    x=np.zeros((n))
    y=np.zeros((n))
    k=np.random.choice(len(pi),n,p=pi,replace=True)
    for i in range(0,len(k)):
        x[i],y[i]=np.random.multivariate_normal(mu[k[i]],cov[k[i]],1).T
    return x,y



