import network_simulator as ns
import numpy as np
import matplotlib.pyplot as plt


# import Scattering_matrix as sm
Quarter_wave = ns.Circuit('Quarter wave resonator')
resonator0 = (ns.Resonator('Resonator_0')
             .add_short('S2')
             # .add_open('O2')
             .add_coupling_capacitor('C2', c=3000e-15)
             .add_transmission_line('S2', 0, 'C2', 0, z0=50, phi0=1e8, length=4e-3))

Quarter_wave.add_resonator(resonator0)
Quarter_wave.finish(starting_position=-10e-3)
kvals = np.linspace(0,1000,1000)
mode_vals = list(map(lambda k: resonator0.mode_condition(k),kvals))
plt.plot(kvals,np.real(mode_vals))
plt.plot(kvals,np.imag(mode_vals))
plt.figure()
plt.plot(kvals,np.abs(mode_vals))
plt.plot(kvals,np.angle(mode_vals))
# plt.show()
print('final value', resonator0.get_eigenvalue())

zvals = np.linspace(-11e-3,-5e-3,100, dtype=np.complex128)
plt.figure()
# print(np.absolute(resonator_eigenfunction(zvals)))
resonator_eigenfunction = resonator0.eigenfunction()
plt.plot(zvals, np.real(resonator_eigenfunction(zvals)))
plt.plot(zvals, np.imag(resonator_eigenfunction(zvals)))
plt.show()