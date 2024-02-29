from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt



class PurcellFilter(Circuit):
    def __init__(self, name='quarter_wave_with_airbridges'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 29.68e-15
        resonator_length = 4000e-6
        airbridge_capacitance = 1e-15
        airbridge_number = 500
        short_position = 0

        self.circuit_elements = {
            'C_o': {
                'element':'Capacitor',
                'values':{
                    'capacitance':capacitance,
                    'position': resonator_length
                }
            },


            'S1': {'element': 'Short',
                   'values': {
                       'position': short_position,
                       }
                    },
        }

        for i in range(airbridge_number):
            self.circuit_elements.update({
                f'C{i}': {
                    'element': 'Capacitor',
                    'values': {
                        'capacitance':airbridge_capacitance,
                        'position': resonator_length/(airbridge_number+1)*(i+1),
                    }
                }
            }
            )

    def define_resonators(self):
        impedance = 50
        phase_velocity = 1e8
        airbridge_number = 5
        self.resonators = {
            'R1': {
                'Short_Airbridge': {
                    'start_pin': {
                        'element': 'S1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'C0',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'transmission_line': {
                        'characteristic_impedance': impedance,
                        'phase_velocity': phase_velocity,
                    },
                },

                'Airbridge_Capacitor':{
                    'start_pin': {
                        'element':f'C{airbridge_number-1}',
                        'side':'main',
                        'pin':'beta',
                    },
                    'end_pin':{
                        'element':'C_o',
                        'side':'main',
                        'pin':'beta',
                    },
                    'transmission_line':{
                        'characteristic_impedance':impedance,
                        'phase_velocity':phase_velocity,
                    },
                },
            }
        }

        for i in range(airbridge_number-1):
            self.resonators['R1'].update({
                f'Airbridge{i}_Airbridge{i+1}':{
                    'start_pin': {
                        'element':f'C{i}',
                        'side':'main',
                        'pin':'beta',
                    },
                    'end_pin':{
                        'element':f'C{i+1}',
                        'side':'main',
                        'pin':'alpha',
                    },
                    'transmission_line':{
                        'characteristic_impedance':impedance,
                        'phase_velocity':phase_velocity,
                    },
                },
            }
            )

if __name__ == '__main__':
    phase_velocity = 1e8
    two_channel_resonator = PurcellFilter('quarter_wave')
    two_channel_resonator.initialize()
    R1 = two_channel_resonator.resonator_dict['R1']

    eigv = R1.get_eigenvalue()
    eigfunc = R1.get_eigenfunction()
    eigfunc.plot()
    plt.show()
    print(eigv)
    
