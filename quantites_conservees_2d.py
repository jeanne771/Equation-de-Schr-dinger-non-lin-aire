#####SPLIT STEP 2 DIMENSIONS

import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2
import cmath

##Introduction des paramètres
kap = 0.033
g = 9.81
w = np.sqrt(g*kap)

theta = np.pi/8 #angle entre les vagues
k = np.cos(theta)
l = np.sin(theta)
cx = k/2
cy = l/2
alpha = (2*l**2-k**2)/8
beta = (2*k**2-l**2)/8
gamma = -3*l*k/4
xi = 1/2
zeta = (k**5-k**3*l**2-3*k*l**4-2*k**4+2*k**2*l**2+2*l**4)/(2*(k-2))

N = 256
Lx = 80
Ly = 200
hx=Lx/N #pas en x
hy=Ly/N
n=np.arange(-N/2, N/2)
X=n*hx
Y=n*hy
frames=10 #nombre d'images
Af=np.zeros([frames*N,N], dtype=complex)
Bf=np.zeros([frames*N,N], dtype=complex)

ind=np.arange(N)
ind[int(N/2)+1:N]-=N
kx=ind*2*np.pi/Lx
ky=ind*2*np.pi/Ly

def partial_x(A):
    [ma,na]=np.shape(A)
    A1=np.zeros([ma, na], dtype=complex)
    for i in range (na-1):
        A1[:,i]= A[:, i+1]
    dax = (A1 - A)/hx
    dax[:,na-1]=dax[:,0]
    return(dax)

def partial_y(A):
    [ma,na]=np.shape(A)
    A1=np.zeros([ma, na], dtype=complex)
    for i in range (ma-1):
        A1[i,:]= A[i+1, :]
    day = (A1 - A)/hy
    day[ma-1,:]=day[0,:]
    return(day)

##Split-step
a=np.zeros([N,N], dtype=complex)
A=np.array(a)
b=np.zeros([N,N], dtype=complex)
B=np.array(b)
a2=np.zeros([N,N], dtype=complex)
A2=np.array(a2)
b2=np.zeros([N,N], dtype=complex)
B2=np.array(b2)

X_grid, Y_grid = np.meshgrid(X, Y)
eta=np.zeros([N, N])

TAU=[0.01,0.1,0.5,1]
for s in range(4):
    dt = TAU[s]
    T=20
    M = int(T/dt)
    F=np.floor(M/frames)

    A0=0.1*np.ones((N,N))+0.5*10**(-2)*np.random.uniform(0,1,(N, N))
    B0=0.1*np.ones((N,N))+0.5*10**(-3)*np.random.uniform(0,1,(N, N))
    A=A0
    B=B0

    I=np.zeros(M+1)
    I[0]=np.sum(np.abs(partial_x(A))**2+np.abs(partial_y(A))**2+np.abs(partial_x(B))**2+np.abs(partial_y(B))**2 + (xi/2)*(np.abs(A)**4 + np.abs(B)**4)+2*zeta*(np.abs(A)**2)*(np.abs(B)**2))

    IA=np.zeros(M+1)
    IB=np.zeros(M+1)

    IA[0]=np.sum(np.abs(A)**2)*hx*hy
    IB[0]=np.sum(np.abs(B)**2)*hx*hy

    for m in range(1, M+1):
        A0 = A
        A= A*np.exp(-1j*dt*(xi*np.abs(A)**2+2*zeta*np.abs(B)**2))
        B= B*np.exp(-1j*dt*(xi*np.abs(B)**2+2*zeta*np.abs(A0)**2))
        A2=np.fft.fft2(A)
        B2=np.fft.fft2(B)
        A2=A2*np.exp(-1j*(cx*kx+cy*ky+alpha*kx**2+beta*ky**2+gamma*ky*kx)*dt)
        B2=B2*np.exp(-1j*(cx*kx-cy*ky+alpha*kx**2+beta*ky**2-gamma*ky*kx)*dt)
        A=np.fft.ifft2(A2)
        B=np.fft.ifft2(B2)

        #quantitees conservees
        I[m]=np.sum(np.abs(partial_x(A))**2+np.abs(partial_y(A))**2+np.abs(partial_x(B))**2+np.abs(partial_y(B))**2 + (xi/2)*(np.abs(A)**4 + np.abs(B)**4)+2*zeta*(np.abs(A)**2)*(np.abs(B)**2))
        IA[m]=np.sum(np.abs(A)**2)*hx*hy
        IB[m]=np.sum(np.abs(B)**2)*hx*hy

        #enregistrement pour figures
        if m%F == 0 and m//F<=frames :
            k=int(m//F*N)
            Af[k-N:k,:]=A
            Bf[k-N:k,:]=B
           

    ##figures
    t=np.linspace(0, T, M+1)
    J=np.ones(M+1)*I[0]
    K=(I-J)/I[0]
    JA=np.ones(M+1)*IA[0]
    KA=(IA-JA)/IA[0]
    JB=np.ones(M+1)*IB[0]
    KB=(IB-JB)/IB[0]

    plt.figure('Perte 2 energie 2d')
    plt.plot(t,K,label='tau={:}'.format(dt))
    plt.xlabel('t')
    plt.ylabel('(I(t) - I(0)) / I[0]')
    plt.title('Perte énergie 2D')
    plt.legend()

    plt.figure('Perte energie 2d')
    plt.plot(t,KA,label='A, tau={:}'.format(dt))
    plt.plot(t,KB,label='B, tau={:}'.format(dt))
    plt.xlabel('t')
    plt.ylabel('(I(t) - I(0)) / I[0]')
    plt.title('Perte énergie 2D')
    plt.legend()

plt.show()