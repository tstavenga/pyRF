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
        capacitance = 30e-15
        open_position = 0
        resonator_length = 4e-3

        self.circuit_elements = {
            'O1': {
                'element': 'Open',
                'values': {
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
                        'element': 'O1',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'S1',
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
    quarter_wave_circuit.resonator_dict['R1'].scattering_matrix(500)
    # for resonator_name, resonator in quarter_wave_circuit.resonator_dict.items():
        # resonator.scattering_matrix(5)
    bla = 8
    # quarter_wave_circuit.
