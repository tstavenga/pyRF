import network_simulator as ns
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

feedlineCircuit = ns.Circuit('Feedline Circuit')


feedline = (ns.SignalPath('Feedline')
                      .add_matched_load('R1')
                      .add_matched_load('R2')
                      .add_reflector('BS1')
                      .add_reflector('BS2')
                      .add_reflector('BS3')
                      .add_transmission_line('R1', 0, 'BS1', 0, z0=50, phi0=1e8, length=150e-3)
                      .add_transmission_line('BS1', 0, 'BS2', 0, z0=50, phi0=1e8, length=10e-3)
                      .add_transmission_line('BS2', 0, 'BS3', 0, z0=50, phi0=1e8, length=10e-3)
                      .add_transmission_line('BS3', 0, 'R2', 0, z0=50, phi0=1e8, length=140e-3))
feedlineCircuit.add_signal_path(feedline)
feedlineCircuit.finish()

# print(feedline.eigenfunctions)
# feedlineFunction = feedline.get_eigenfunction(100)
zvals = np.linspace(0e-3,300e-3,1500)
yright = np.exp(-(zvals-145e-3)**2/0.6e-3**2)
# plt.figure()
# plt.plot(zvals,yright)
# plt.show()
# for phi in feedline.eigenfunctions(500,zvals):
    # print(phi)
    # plt.plot(zvals,np.abs(phi))
# plt.show()
# k = np.fft.fftfreq(1000,d=100e10/1)
# for i in k:
# for phi in feedline.eigenfunction_coefficients_g(500):
#     print(phi)
# print(list(map(lambda x: next(feedline.eigenfunction_coefficients(x)),k)))
# print(list(map(lambda x: next(feedline.eigenfunction_coefficients(x)),k)))
# phi1, phi2 = list(feedline.eigenfunction_coefficients_g(500))
# print(phi1.flatten().dot(np.conjugate(phi1.flatten())))
# eigs = feedline.eigenfunction_coefficients_v(k)
# print(len(eigs))
# eig1 = eigs[0]
# eig2 = eigs[1]
# # print(eig1[1,1,:])
# plt.plot(eig1[0,:])
# plt.plot(eig2[0,:])
# plt.show()
sol = feedline.solution(zvals,yright,yright*0, t=0.6,c=10e-3)
plt.figure()
for s in sol:
    plt.plot(zvals,abs(s))
plt.figure()
k2 = np.linspace(500,500+1e-10,100)
phi = feedline.eigenfunction_coefficients_v(k2)
plt.plot(np.arange(len(k2)),np.real(phi[0,0,:]))
plt.plot(np.arange(len(k2)),np.real(phi[1,0,:]))


# phi = feedline.eigenfunction_coefficients(500)
# Aind = np.argmax(np.abs(phi[0,:]))
# A = phi[:,Aind]/phi[0,Aind]
# Bind = np.argmax(np.abs(phi[-1,:]))
#
# B = phi[:,Bind]
# B = B-A*B[0]
# B = B/B[-1]
# A = A-B*A[-1]
#
# print(np.array([A,B]).T)
#
# print('A',A)
# print('B',B)
# print(phi)

# print(phi[0,0,:])
# print(phi[0,0,:])
feedline.solution_animation(zvals,yright*0,yright, times=np.arange(0,30,0.1),c=20e-3)
plt.show()
#
# smat = np.array([[-1., 0.70710678, 0.70710678, 0],
#                  [0., 0., 0., 0.],
#                  [0., 0., 0., 0., ],
#                  [0., 0.70710678, 0.70710678, -1.]])
# from scipy.linalg import orth
# # smat = np.array([[0., 0., 1., 0],
# #                  [0., 1., 0., 0.],
# #                  [0., 0., 1., 0., ],
# #                  [0., 1., 0., 0.]])
# smat = np.array([[1., 0., 0., 0.],
#                  [-np.sqrt(2)/2, 0., np.sqrt(2)/2, 0.],
#                  [np.sqrt(2)/2, 0., -np.sqrt(2)/2, 0.],
#                  [0., 0., 1., 0.]])
# print(orth(smat))
# phi = orth(smat)
# phi1 = phi[:,0]
# phi2 = phi[:,1]
# print(phi1)
# print(phi2)
# zleft = np.linspace(-10e-3,0,100)
# zright = np.linspace(0,10e-3,100)
# k=500
# print(phi1[0:2])
# print(phi1[2:4])
# phi1left = np.dot(phi1[0:2],[np.exp(1j*k*zleft), np.exp(-1j*k*zleft)])
# phi1right = np.dot(phi1[2:4],[np.exp(1j*k*zright), np.exp(-1j*k*zright)])
# # k=-500
# phi2left = np.dot(phi2[0:2],[np.exp(1j*k*zleft), np.exp(-1j*k*zleft)])
# phi2right = np.dot(phi2[2:4],[np.exp(1j*k*zright), np.exp(-1j*k*zright)])
#
# plt.plot(zleft,np.abs(phi1left), color='tab:orange')
# plt.plot(zright,np.abs(phi1right), color='tab:orange')
# plt.plot(zleft,np.abs(phi2left), color='tab:blue')
# plt.plot(zright,np.abs(phi2right), color='tab:blue')
