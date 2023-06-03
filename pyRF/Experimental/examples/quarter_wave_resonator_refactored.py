from pyRF import Circuit
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
        CAPACITANCE = 30e-15
        CAPACITOR_POSITION = 0
        RESONATOR_LENGTH = 4e-3

        self.circuit_elements = {
            'C1': {'element': 'GroundedCapacitor',
                    'options': {
                        'capacitance': CAPACITANCE,
                        'position': CAPACITOR_POSITION,
                        }
                    },

            'S1': {'element': 'Short',
                   'options': {
                       'position': RESONATOR_LENGTH,
                       }
                    },
        }

    def define_transmission_lines(self):
        IMPEDANCE = 50
        PHASE_VELOCITY = 1e8

        self.transmission_lines = {
            'T1': {'impedance': IMPEDANCE,
                   'phase_velocity': PHASE_VELOCITY}
        }

    def define_resonator(self):
        self.resonators = {
            'R1': {}
        }
if __name__ == '__main__':
    quarter_wave_circuit = QuarterWave('quarter_wave')
