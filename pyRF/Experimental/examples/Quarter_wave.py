from pyRF import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
from pyRF.scattering_matrix import TransmissionLine
import numpy as np
import matplotlib.pyplot as plt


# import Scattering_matrix as sm
Quarter_wave = Circuit('Quarter wave resonator')

resonator_length = 4e-3
capacitance = 30e-15

Z0 = 50
phi = 1e8

capacitor_1 = ne.GroundedCapacitor('C1',
                                   position = resonator_length,
                                   capacitance = capacitance)

short_1 = ne.Short('S1',
                   position = 0)

line_1 = TransmissionLine(start = 'S1',
                          end = 'C1',
                          Z0 = Z0,
                          phi = phi)

Quarter_wave.add_elements(capacitor_1, short_1)

resonator_0 = Resonator('resonator_0')
resonator_0.add_circuit_elements(capacitor_1, short_1)
resonator_0.add_transmission_lines(line_1)


resonator_0.add_transmission_lines()
             .add_short('S2')
             # .add_open('O2')
             .add_coupling_capacitor('C2', c=30e-15)
             .)

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
resonator_eigenfunction = resonator0.get_eigenfunction()
plt.plot(zvals, np.real(resonator_eigenfunction(zvals)))
plt.plot(zvals, np.imag(resonator_eigenfunction(zvals)))
plt.show()