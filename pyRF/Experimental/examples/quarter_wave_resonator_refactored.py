from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
from pyRF.scattering_matrix import TransmissionLine
import numpy as np
import matplotlib.pyplot as plt


quarter_wave_circuit = Circuit('Quarter wave resonator')

class QuarterWave(Circuit):
    def __init__(self, name='Default'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 30e-15
        capacitor_position = 0
        resonator_length = 4e-3

        self.circuit_elements = {
            'C1': {
                'element': 'GroundedCapacitor',
                'options': {
                    'capacitance': capacitance,
                    'position': capacitor_position,
                }
            },

            'S1': {'element': 'Short',
                   'options': {
                       'position': resonator_length,
                       }
                    },
        }

    def define_transmission_lines(self):
        impedance = 50
        phase_velocity = 1e8

        self.transmission_lines = {
            'T1': {
                'impedance': impedance,
                'phase_velocity': phase_velocity}
        }

    def define_resonator(self):

        self.resonators = {
            'R1': {
                'Capacitor_Short': {
                    'start_pin': {
                        'element': 'C1',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'S1',
                        'pin': 'alpha',
                    },
                    'transmission_line': 'T1',
                }
            }
        }


if __name__ == '__main__':
    quarter_wave_circuit = QuarterWave('quarter_wave')
    quarter_wave_circuit.initialize()
