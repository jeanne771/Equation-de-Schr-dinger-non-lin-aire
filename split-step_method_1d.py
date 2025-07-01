import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2
import cmath

#introduction des paramètres
tau=0.09 #pas de temps
gamma=2
T= 36
M=int(T/tau) #nombre de points en temps
t=np.linspace(0, T, M)

#Discrétisation de l'espace
L=16 # période de la fonction considérée
sigma=2*np.pi/L
N=30 # nombre de points en espace
h=L/N # pas d'espace
n=np.arange(-N/2, N/2)
X=n*h

#condition initiale
a=0.5
u0=lambda x: a*(1+(0.1*(1-2*np.abs(X)/L)))

U=np.zeros([M,N], dtype=complex)
V=np.zeros([M,N], dtype=complex)
U2=np.zeros([M,N], dtype=complex)
V2=np.zeros([M,N], dtype=complex)

U[0,:]=u0(X)
V[0, :]=-4*n**2*np.pi**2/(L**2)

##Split-step method pour résoudre l'équation de Schrödinger non linéaire, en utilisant Fourier

for i in range(M-1):
  for k in range(N):
    V[i+1,k]=cmath.exp(1j*tau*gamma*np.abs(U[i,k])**2)*U[i,k]
  V2[i+1,:]=np.fft.fftshift(np.fft.fft(V[i+1,:]))
  for k in range(N):
    U2[i+1,k]=cmath.exp(-1j*(k-N/2)**2*sigma**2*tau)*V2[i+1,k]
  U[i+1,:]=np.fft.ifft(np.fft.fftshift(U2[i+1,:]))


#Création d'une figure en 3D
plt.figure("Représentation 3D de la solution")
axes = plt.axes(projection="3d")
X_grid, t_grid = np.meshgrid(X, t)
axes.plot_surface(X_grid, t_grid, np.abs(U), cmap=cm.coolwarm)

axes.set_xlabel("X")
axes.set_ylabel("t")
axes.set_zlabel("|u(x,t)|")

plt.show()