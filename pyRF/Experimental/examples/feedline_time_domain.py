import numpy as np
import matplotlib.pyplot as plt
from pyRF.circuit import Circuit
from pyRF.feedline import TimeDomainSolution





class Feedline(Circuit):
    def __init__(self, name='time domain feedline'):
        super().__init__(name)

    def define_circuit_elements(self):
        feedline_capacitance = 16.03e-15*0
        capacitor_position = 10e-3
        feedline_start = 0
        feedline_end = 20e-3
        port_impedance = 50 

        self.circuit_elements = {
            'P1': {
                'element': 'Port',
                'values': {
                    'position': feedline_start,
                    'impedance': port_impedance,
                }
            },
            'P2': {
                'element': 'Port',
                'values':{
                    'position': feedline_end,
                    'impedance': port_impedance,
                }
            },
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':feedline_capacitance,
                    'position': capacitor_position,
                }
            },

        }

    def define_feedlines(self):
        self.feedlines = {
            'F1': {
                'Port1_Capacitor': {
                    'start_pin': {
                        'element': 'P1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'C1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'transmission_line': {
                        'characteristic_impedance': 50,
                        'phase_velocity': 1e8,
                    },
                },
                'Capacitor_Port2': {
                    'start_pin': {
                        'element': 'C1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'P2',
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
    feedline_circuit = Feedline()
    feedline_circuit.initialize()
    F1 = feedline_circuit.feedline_dict['F1']
    E1, E2 = F1.get_eigenfunction(50)
    fig, ax = plt.subplots(1,1)
    E1.plot(ax, color='tab:blue')
    E2.plot(ax, color='tab:orange')
    plt.show()
    z = np.linspace(0,20e-3,200)
    solution = TimeDomainSolution(F1, z)
    print(solution)