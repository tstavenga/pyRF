from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt


quarter_wave_circuit = Circuit('Quarter wave resonator')

class QuarterWave(Circuit):
    def __init__(self, name='Default'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 300000e-15
        open_position = 0
        resonator_length = 4e-3

        self.circuit_elements = {
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':capacitance,
                    'position': open_position,
                }
            },

            'S1': {'element': 'Short',
                   'values': {
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
                        'element': 'S1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'transmission_line': {
                        'characteristic_impedance': 50,
                        'phase_velocity': 1e8,
                    },
                }
            }
        }

        


if __name__ == '__main__':
    quarter_wave_circuit = QuarterWave('quarter_wave')
    quarter_wave_circuit.initialize()
    R1 = quarter_wave_circuit.resonator_dict['R1']
    print(R1.scattering_matrix(1.354+1j*1e-6))
    E1 = R1.get_eigenvalue()
    E2 = R1.get_eigenvalue(n=2)

    print('E1', E1)
    print('E2', E2)
    ks = np.linspace(0,200,20001,dtype=np.complex128)
    mc = np.array(list(map(lambda k:R1.mode_condition(k), ks)))
    # print(mc)
    plt.plot(ks,mc)
    plt.plot(E1,0, 'x')
    plt.plot(E2,0, 'x')
    plt.show()
    # for resonator_name, resonator in quarter_wave_circuit.resonator_dict.items():
        # resonator.scattering_matrix(5)
    bla = 8
    # quarter_wave_circuit.
