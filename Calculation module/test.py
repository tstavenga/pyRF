import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

u = np.linspace(0,2*np.pi,100)
v = np.linspace(0,np.pi,100)

pow = 0.2

cu = np.sign(np.cos(u))*np.abs(np.cos(u))**pow
su = np.sign(np.sin(u))*np.abs(np.sin(u))**pow
cv = np.sign(np.cos(v))*np.abs(np.cos(v))**pow
sv = np.sign(np.sin(v))*np.abs(np.sin(v))**pow

x = 10*np.outer(cu,sv)
y = 10*np.outer(su,sv)
z = 10*np.outer(np.ones(len(u)),cv)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x,y,z,antialiased=False)
plt.show()