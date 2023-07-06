import network_simulator as ns
import numpy as np
import matplotlib.pyplot as plt

resonator_feedline_circuit = ns.Circuit('Resonator Feedline')


c = 30e-16
phi0 = 1e8
z0 = 50
clengthinv = phi0*z0
resonator = (ns.Resonator('Resonator 1')
                      .add_short('S1')
                      .add_coupling_capacitor('C1', c=c)
                      .add_transmission_line('C1', 0, 'S1', 0, z0=z0, phi0=phi0, length=4000e-6))
resonator_feedline_circuit.add_resonator(resonator)
feedline = (ns.SignalPath('Feedline')
                      .add_matched_load('R1')
                      .add_matched_load('R2')
                      .add_transmission_line('R1', 0, 'C1', 1, z0=z0, phi0=phi0, length=10e-3)
                      .add_transmission_line('R2', 0, 'C1', 1, z0=z0, phi0=phi0, length=10e-3))
resonator_feedline_circuit.add_signal_path(feedline)
resonator_feedline_circuit.finish()
k = resonator.get_eigenvalue()
resonator.normalize_eigenfunction()
print('norm')
resonator_eigenfunction = resonator.eigenfunction
# print(k)

kvals = np.linspace(0,200,200)
mode_vals = list(map(lambda k: resonator.mode_condition(k),kvals))
plt.figure()
# print(np.array(mode_vals)[:,0] + 1j*np.array(mode_vals)[:,1])
# plt.plot(kvals,np.abs(np.array(mode_vals)[:,0] + 1j*np.array(mode_vals)[:,1]))
# plt.plot(kvals,np.real(mode_vals))
# plt.plot(kvals,np.imag(mode_vals))
# plt.show()
resonator.plot_eigenfunction()
# plt.show()
phi = next(feedline.eigenfunctions(50,np.array([10e-3],dtype=np.complex128)))[0]
print('phi',phi)
psi = resonator_eigenfunction(0)
print('psi',psi)
# pos = ns.scattering_matrix_dict['C1'].get_position()
# print(pos)
jflk = phi*c*clengthinv*psi
jresk = phi*c*clengthinv*psi

TDcoupling = jflk*jresk*np.pi**2/phi0*2

print(TDcoupling)
print(np.roots([1j*abs(TDcoupling),-1,0,phi0**2*k**2]))
print(k)
print(k*phi0)
frescomp = np.roots([1j*abs(TDcoupling),-1,0,phi0**2*k**2])[1]
frescomp1, frescomp2 = np.roots([1j*abs(TDcoupling),-1,0,phi0**2*k**2])[1:]
print('f1',frescomp1)
print('f2',frescomp2)
fdrive = np.linspace(np.real(frescomp)-500e0, np.real(frescomp)+500e0,20000,dtype=np.clongdouble)
C = -np.abs(jflk)*1j*fdrive**0/((fdrive-frescomp1))#*(fdrive+frescomp2))
print('fres',np.real(frescomp))
# print('fres',frescomp**2)
plt.figure()
plt.plot(fdrive,np.abs(C))
plt.plot(fdrive,np.real(C))
plt.plot(fdrive,np.imag(C))
A = 2-1*np.abs(jresk)*C*fdrive**2*np.pi**2/phi0
B = -jresk*C*fdrive*np.pi**2/phi0
plt.figure()
# plt.plot(fdrive,np.real(A))
# plt.plot(fdrive,np.imag(A))
# plt.plot(fdrive,np.abs(A))
plt.plot(np.real(A),np.imag(A))
# plt.yscale('log')
plt.title('A')
# plt.plot(fdrive,np.real(C))
# plt.plot(np.real(C),np.imag(C))
# plt.show()
fdrive = np.real(frescomp)
C2 = np.abs(jflk)*fdrive**1/((fdrive-frescomp1)*(fdrive+frescomp2))
print('C2',C2)
print('Q',np.real(frescomp)/np.imag(frescomp))
print('C2/Q',C2/np.real(frescomp)*np.imag(frescomp))


# plt.show()
cs = np.linspace(10e-15,100e-15,20)
qs = []
for c in cs:
    jflk = phi*c*clengthinv*psi
    jresk = phi*c*clengthinv*psi

    TDcoupling = np.pi*jflk*jresk/2/phi0
    qs.append(np.max(np.imag(np.roots([2j * np.pi * abs(TDcoupling), -1, 0, phi0 ** 2 * k ** 2]))))
plt.figure()
plt.plot(cs,np.array(qs))
plt.show()
# zvals = np.linspace(0,4414e-6,100, dtype=np.complex128)
# fig,ax = plt.subplots(1,1)
# # print(np.absolute(resonator_eigenfunction(zvals)))
# plt.plot(zvals, np.absolute(resonator_eigenfunction(zvals)))
# plt.show()
# figf,axf = plt.subplots(1,1)
# zfeedline = np.linspace(0,20e-3,200, dtype=np.complex128)
# feedline_eigenfunction = feedline.get_eigenfunction(k)
# axf.plot(zfeedline, np.absolute(feedline_eigenfunction(zfeedline)))

# plt.show()
# print(ns.element_dict)

                      # .finish())
# resonator_feedline.add_capacitor('C1',c=10e-15)
# space_graph = resonator_feedline.space_network

# print(cap._space_ports)