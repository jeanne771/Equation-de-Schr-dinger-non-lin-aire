#####SPLIT STEP 2 DIMENSIONS
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

#intervalle de temps entre deux images (en seconde)
PAUSE=1e-1

##Introduction des paramètres (physique)
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


N = 256 #nombre de points en espace
dt = 0.01 #pas de temps
M = 1000 #Nombre de pas de temps
T=M*dt
Lx = 80
Ly = 200
hx=Lx/N #pas en x
hy=Ly/N #pas en y
n=np.arange(-N/2, N/2)
X=n*hx
Y=n*hy
frames=10 #nombre d'images
F=np.floor(M/frames)
Af=np.zeros([frames*N,N], dtype=complex)
Bf=np.zeros([frames*N,N], dtype=complex)

ind=np.arange(N)
ind[int(N/2)+1:N]-=N
kx=ind*2*np.pi/Lx
ky=ind*2*np.pi/Ly

#conditions initiales
A0=0.1*np.ones((N,N))+0.5*10**(-2)*np.random.uniform(0,1,(N, N))
B0=0.1*np.ones((N,N))+0.5*10**(-2)*np.random.uniform(0,1,(N, N))

##Split-step method
a=np.zeros([N,N], dtype=complex)
A=np.array(a)
b=np.zeros([N,N], dtype=complex)
B=np.array(b)
a2=np.zeros([N,N], dtype=complex)
A2=np.array(a2)
b2=np.zeros([N,N], dtype=complex)
B2=np.array(b2)

A=A0
B=B0

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

    #enregistrement pour figures
    if m%F == 0 :
        k=int(m//F*N)
        Af[k-N:k,:]=A
        Bf[k-N:k,:]=B
        


##Figure
plt.ion()
plt.show()
plt.figure("Visualisation 3D")
axes = plt.axes(projection="3d")


X_grid, Y_grid = np.meshgrid(X, Y)
eta=np.zeros([N, N])

for i in range(frames):
    plt.figure("Visualisation 3D")
    axe = plt.axes(projection="3d")
    
    #Affichage de la surface d'élévation de la mer (eta)
    t=(i+1)*F*dt
    eta= np.real((Af[i*N:(i+1)*N,:]/kap)*np.exp(1j*(k*X_grid+l*Y_grid-t))+(Bf[i*N:(i+1)*N,:]/kap)*np.exp(1j*(k*X_grid-l*Y_grid-t)))
    axe.plot_surface(X_grid/kap, Y_grid/kap, eta, cmap=cm.coolwarm)
    axe.set_xlabel("X")
    axe.set_ylabel("Y")
    axe.set_zlabel("Surface d'élévation")
    plt.title('t={:.2f}'.format(t))
    plt.pause(PAUSE)
    plt.clf()

#Même animation mais zoomée
for i in range(frames):
    plt.figure('Visualisation 3D zoom')
    axe = plt.axes(projection="3d")

    t=(i+1)*F*dt
    Ap=Af[(i-1)*N+112:i*N-112,112:144]
    Bp=Bf[(i-1)*N+112:i*N-112,112:144]
    x_grid=X_grid[112:144,112:144]
    y_grid=Y_grid[112:144,112:144]

    #Affichage de la surface d'élévation de la mer (eta)
    eta= np.real((Ap/kap)*np.exp(1j*(k*x_grid+l*y_grid-t))+(Bp/kap)*np.exp(1j*(k*x_grid-l*y_grid-t)))
    axe.plot_surface(x_grid/kap, y_grid/kap, eta, cmap=cm.coolwarm)
    axe.set_xlabel("X")
    axe.set_ylabel("Y")
    axe.set_zlabel("Surface d'élévation")
    plt.title('t={:.2f}'.format(t))
    plt.pause(PAUSE)
    plt.clf()