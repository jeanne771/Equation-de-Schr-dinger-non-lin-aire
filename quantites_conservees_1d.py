##Verification des quantites conservees

import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2
import cmath

#Discrétiation de l'espace
L=16
sigma=2*np.pi/L
N=30
h=L/N
n=np.arange(-N/2, N/2)
X=n*h
CFL=h**2 / np.pi

#condition initiale
alpha=1
gamma=2
a=0.5
u0=lambda x: a*(1+(0.1*(1-2*np.abs(X)/L)))

def partial_x(A):
    [ma,na]=np.shape(A)
    A1=np.zeros([ma, na], dtype=complex)
    for i in range (na-1):
        A1[:,i]= A[:, i+1]
    dax = (A1 - A)/h
    dax[:,na-1]=dax[:,0]
    return(dax)

TAU=[0.5,0.9,0.99,1.01]

for s in range(4) :
    tau=TAU[s]*CFL
    T= 10
    M=np.floor(T/tau)
    M=int(M)

    U=np.zeros([M,N], dtype=complex)
    V=np.zeros([M,N], dtype=complex)
    U2=np.zeros([M,N], dtype=complex)
    V2=np.zeros([M,N], dtype=complex)

    U[0,:]=u0(X)
    V[0, :]=-4*n**2*np.pi**2/(L**2)

    for i in range(M-1):
        for k in range(N):
            V[i+1,k]=cmath.exp(1j*tau*gamma*np.abs(U[i,k])**2)*U[i,k]
            V2[i+1,:]=np.fft.fftshift(np.fft.fft(V[i+1,:]))
        for k in range(N):
            U2[i+1,k]=cmath.exp(-1j*(k-N/2)**2*sigma**2*tau)*V2[i+1,k]
            U[i+1,:]=np.fft.ifft(np.fft.fftshift(U2[i+1,:]))

    #Verification des quantités conservées / module

    I1 = np.zeros(M)
    for i in range(M):
        for k  in range(N):
            I1[i]+=np.abs(U[i,k])**2
    I1=h*I1
    J1=np.ones(M)*I1[0]
    K1=(I1-J1)/I1[0]

    E = np.zeros([M,N], dtype=complex)
    E = 1/2 * alpha * np.abs(partial_x(U))**2 - 1/4 * gamma * np.abs(U)**4
    I2=np.zeros(M)
    for i in range (M) :
        I2[i]=np.sum(E[i,:])
    I2=h*I2
    J2=np.ones(M)*I2[0]
    K2=(I2-J2)/I2[0]

    #représentation visuelle
    t=np.linspace(0, T, M)

    plt.figure("perte module")
    plt.plot(t,K1,label='tau/CSN={:}'.format(TAU[s]))
    plt.xlabel('t')
    plt.ylabel('I(t) - I(0)/I(0)')
    plt.title('condition de stabilité numérique h²/π={:.3f}'.format(CFL))
    plt.legend()

    plt.figure("perte énergie")
    plt.plot(t,K2,label='tau/CSN={:}'.format(TAU[s]))
    plt.xlabel('t')
    plt.ylabel('I(t) - I(0)')
    plt.title('condition de stabilité numérique h²/π={:.3f}'.format(CFL))
    plt.legend()

plt.show()