from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt
import cProfile

PHASE_VELOCITY = 124998795.11524159
LENGTH = 8.095835673090729e-3

class QuarterWave(Circuit):
    def __init__(self, name='Quarter wave', phase_velocity = PHASE_VELOCITY, length = LENGTH):
        super().__init__(name)
        self.phase_velocity = phase_velocity
        self.length = length

    def define_circuit_elements(self):
        qubit_resonator_capacitance = 16.03e-15
        feedline_resonator_capacitance = 5.69e-15
        # qubit_resonator_capacitance = 33e-15
        # feedline_resonator_capacitance = 22.45e-15
        qubit_position = 0
        resonator_length = self.length

        self.circuit_elements = {
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':qubit_resonator_capacitance,
                    'position': qubit_position,
                }
            },

            'C2': {'element': 'Capacitor',
                   'values': {
                       'capacitance':feedline_resonator_capacitance,                       
                       'position': resonator_length,
                       }
                    },
        }


    def define_resonators(self):
        self.resonators = {
            'R1': {
                'Capacitor_Short': {
                    'start_pin': {
                        'element': 'C1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'C2',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'transmission_line': {
                        'characteristic_impedance': 50,
                        'phase_velocity': self.phase_velocity,
                    },
                }
            }
        }

        


if __name__ == '__main__':
    pv = np.array([122333454.18287343,
                   122407024.01080994,
                   122507509.51707558,
                   122450681.12521733,
                   ])
    phase_velocity = np.mean(pv)
    print(phase_velocity)
    # phase_velocity = 120151694.27702929
    # phase_velocity = 120313297
    # # phase_velocity = 124965229.10070007
    # # # phase_velocity = 123843123
    length = 0.008364094353082047
    quarter_wave_circuit = QuarterWave('quarter_wave',
                                       phase_velocity=phase_velocity,
                                       length=length)
    quarter_wave_circuit.initialize()
    R1 = quarter_wave_circuit.resonator_dict['R1']
    eigv = R1.get_eigenvalue()
    print(eigv*phase_velocity)


    ####################
    ### length sweep ###
    ####################

    # lt = np.linspace(7.6e-3,8.9e-3,100)
    # eigs = []
    # for length in lt:
    #     quarter_wave_circuit = QuarterWave('quarter_wave', phase_velocity=phase_velocity, length=length)
    #     quarter_wave_circuit.initialize()
    #     R1 = quarter_wave_circuit.resonator_dict['R1']
    #     eigv = R1.get_eigenvalue()
    #     eigs.append(eigv*phase_velocity)

    # eigvs = np.array(eigs)
    # plt.plot(lt*1e3,eigvs, 'x-')
    # freqdes = 7.300e9
    # opt = np.interp(freqdes, eigvs[::-1], lt[::-1])
    # print(opt)
    # plt.plot(opt*1e3,freqdes,'o',markersize=10)
    # plt.show()

    ############################
    ### phase velocity sweep ###
    ############################

    # pv = np.linspace(1.2e8,1.3e8,100)
    # eigs = []
    # for phase_velocity in pv:
    #     print(phase_velocity)
    #     quarter_wave_circuit = QuarterWave('quarter_wave', phase_velocity=phase_velocity,length = 7.5837713e-3)
    #     quarter_wave_circuit.initialize()
    #     R1 = quarter_wave_circuit.resonator_dict['R1']
    #     eigv = R1.get_eigenvalue()
    #     print(eigv*phase_velocity)
    #     eigs.append(eigv*phase_velocity)

    # eigvs = np.array(eigs)
    # plt.plot(pv,eigvs, 'x-')
    # freqmeas = 8.05073740768215e9
    # opt = np.interp(freqmeas, eigvs, pv)
    # print(opt)
    # plt.plot(opt,freqmeas,'o',markersize=10)
    # plt.show()
