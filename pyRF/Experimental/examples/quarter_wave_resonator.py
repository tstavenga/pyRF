from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt
import cProfile


class QuarterWave(Circuit):
    def __init__(self, name='Quarter wave'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 30e-15
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
    eigv = R1.get_eigenvalue()
    print(eigv)

    eig = R1.get_eigenfunction(n=2)
    eig.plot()
    plt.show()