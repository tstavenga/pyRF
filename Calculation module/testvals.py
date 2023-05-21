import network_simulator as ns
import numpy as np
import matplotlib.pyplot as plt

zvals = np.linspace(0,100,100)
k = 0.1
x = 0.5
p2 = np.exp(1j*k*zvals)*(1+1j*x/2)-0.5j*x*np.exp(-1j*k*zvals)
mzvals = np.linspace(-100,0,100)
p1 = np.exp(1j*k*mzvals)
plt.figure()
plt.plot(mzvals,np.abs(p1))
plt.plot(zvals,np.abs(p2))
plt.show()