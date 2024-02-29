from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt



class TwoChannelResonator(Circuit):
    def __init__(self, name='two_channel_resonator', phase_velocity=1e8, characteristic_impedance = 50, coupling_capacitance=29.68e-15, open_capacitance=24.5e-15, open_position=3547.4e-6, capacitor_position=3200e-6, short_position=0.):
        super().__init__(name)
        self.phase_velocity = phase_velocity
        self.coupling_capacitance = coupling_capacitance
        self.open_capacitance = open_capacitance
        self.open_position = open_position
        self.capacitor_position = capacitor_position
        self.short_position = short_position
        self.characteristic_impedance = characteristic_impedance


    def define_circuit_elements(self):

        self.circuit_elements = {
            'C2': {
                'element':'Capacitor',
                'values':{
                    'capacitance':self.open_capacitance,
                    'position': self.open_position
                }
            },
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':self.coupling_capacitance,
                    'position': self.capacitor_position,
                }
            },

            'S1': {'element': 'Short',
                   'values': {
                       'position': self.short_position,
                       }
                    },
        }


    def define_resonators(self):
        impedance = self.characteristic_impedance
        phase_velocity = self.phase_velocity
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
                        'characteristic_impedance': self.characteristic_impedance,
                        'phase_velocity': self.phase_velocity,
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
    eigenfunction = R1.get_eigenfunction()
    eigenfunction.plot()
    plt.show()
    frequency = eigenvalue
    print('Resonator frequency: {:1.5f} GHz'.format(frequency*1e-9))


