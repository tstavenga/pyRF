from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt



class TwoChannelResonator(Circuit):
    def __init__(self, name='two_channel_resonator'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 29.68e-15
        open_capacitance = 24.5e-15
        open_position = 3547.4e-6
        capacitor_position = 3200e-6
        short_position = 0.

        self.circuit_elements = {
            'C2': {
                'element':'Capacitor',
                'values':{
                    'capacitance':open_capacitance,
                    'position': open_position
                }
            },
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':capacitance,
                    'position': capacitor_position,
                }
            },

            'S1': {'element': 'Short',
                   'values': {
                       'position': short_position,
                       }
                    },
        }


    def define_resonators(self):
        impedance = 50
        phase_velocity = 1e8
        self.resonators = {
            'R1': {
                'Open_Capacitor':{
                    'start_pin': {
                        'element':'S1',
                        'side':'main',
                        'pin':'alpha',
                    },
                    'end_pin':{
                        'element':'C1',
                        'side':'main',
                        'pin':'beta',
                    },
                    'transmission_line':{
                        'characteristic_impedance':impedance,
                        'phase_velocity':phase_velocity,
                    },
                },
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
                        'phase_velocity': 1e8,
                    },
                }
            }
        }


if __name__ == '__main__':
    phase_velocity = 1e8
    two_channel_resonator = TwoChannelResonator('quarter_wave')
    two_channel_resonator.initialize()
    R1 = two_channel_resonator.resonator_dict['R1']
    eigenvalue = R1.get_eigenvalue()
    frequency = eigenvalue*phase_velocity
    print('Resonator frequency: {:1.5f} GHz'.format(frequency*1e-9))


