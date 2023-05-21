import network_simulator as ns
import scipy
import scipy.integrate
import mpmath
import numpy as np
import matplotlib.pyplot as plt

resonator_feedline_circuit = ns.Circuit('two channel resonator')
l = 3547.4e-6
shortlength = 3200e-6
resonator = (ns.Resonator('Resonator')
             .add_short('S1')
             # .add_open('O1')
             .add_coupling_capacitor('C2', c=29.68e-15)
             .add_coupling_capacitor('C1', c=54.5e-14) #24.5e-15
             .add_transmission_line('S1', 0, 'C1', 0, z0=50, phi0=100.04e6, length=shortlength)
             .add_transmission_line('C1', 0, 'C2', 0, z0=50, phi0=100.04e6, length=l-shortlength))
resonator_feedline_circuit.add_resonator(resonator)
# resonator_feedline_circuit.add_resonator(resonator0)
resonator_feedline_circuit.finish(starting_position=0)


# print(resonator.guess_phase)
# print(resonator.length)
# print((2 * np.pi - resonator.guess_phase) / abs(2*resonator.length)/6)

kvals = np.linspace(0,200,200)
mode_vals = list(map(lambda k: resonator.mode_condition(k),kvals))
plt.figure()
plt.plot(kvals,np.real(mode_vals))
plt.plot(kvals,np.imag(mode_vals))

print('final value', resonator.get_eigenvalue())
# print(resonator.eigenfunction(1e-3))
resonator.plot_eigenfunction()
# plt.show()
# plt.figure()
# plt.plot(kvals,np.abs(mode_vals))
# plt.plot(kvals,np.angle(mode_vals))
# plt.show()

#
#
zvals = np.linspace(0e-3,l,200, dtype=np.complex128)
plt.figure()
resonator.normalize_eigenfunction()
# print(np.absolute(resonator_eigenfunction(zvals)))
# resonator_eigenfunction = resonator.get_eigenfunction()
print('val', scipy.integrate.quad(lambda y: np.abs(resonator.eigenfunction(y)) ,0 ,l, epsabs = 0)[0])
print(np.trapz(np.abs(resonator.eigenfunction(zvals)),x=zvals))
plt.plot(zvals, np.real(resonator.eigenfunction(zvals)))
plt.plot(zvals, np.abs(resonator.eigenfunction(zvals)))
plt.plot(zvals, np.imag(resonator.eigenfunction(zvals)))
plt.show()

# print(resonator.eigenfunction)
