import network_simulator as ns
import numpy as np
import matplotlib.pyplot as plt

resonator_feedline_circuit = ns.Circuit('Resonator Feedline')



resonator = (ns.Resonator('Resonator 1')
                      .add_short('S1')
                      .add_coupling_capacitor('C1', c=300e-15)
                      .add_transmission_line('C1', 0, 'S1', 0, z0=50, phi0=1e8, length=4414e-6))
resonator_feedline_circuit.add_resonator(resonator)
feedline = (ns.SignalPath('Feedline')
                      .add_matched_load('R1')
                      .add_matched_load('R2')
                      .add_transmission_line('R1', 0, 'C1', 1, z0=50, phi0=1e8, length=10e-3)
                      .add_transmission_line('R2', 0, 'C1', 1, z0=50, phi0=1e8, length=10e-3))
resonator_feedline_circuit.add_signal_path(feedline)
resonator_feedline_circuit.finish()
k = resonator.get_eigenvalue()
resonator_eigenfunction = resonator.get_eigenfunction()
print(k)

zvals = np.linspace(0,4414e-6,100, dtype=np.complex128)
fig,ax = plt.subplots(1,1)
# print(np.absolute(resonator_eigenfunction(zvals)))
ax.plot(zvals, np.absolute(resonator_eigenfunction(zvals)))

figf,axf = plt.subplots(1,1)
zfeedline = np.linspace(0,20e-3,200, dtype=np.complex128)
feedline_eigenfunction = feedline.get_eigenfunction(k)
axf.plot(zfeedline, np.absolute(feedline_eigenfunction(zfeedline)))

plt.show()
# print(ns.element_dict)

                      # .finish())
# resonator_feedline.add_capacitor('C1',c=10e-15)
# space_graph = resonator_feedline.space_network

# print(cap._space_ports)