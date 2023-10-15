from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt



class ResonatorFeedline(Circuit):
    def __init__(self, name='resonator feedline'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 30e-10
        open_position = 0
        resonator_length = 4e-3

        feedline_start = 0
        feedline_capacitor = 10e-3
        feedline_end = 20e-3
        port_impedance = 50.
        self.circuit_elements = {
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':capacitance,
                    'position': {
                        'main': open_position,
                        'coupled': feedline_capacitor,
                    }
                }
            },

            'S1': {
                'element': 'Short',
                'values': {
                    'position': resonator_length,
                }
            },
            'P1': {'element': 'Port',
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
            }
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
                        'side':'coupled',
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
    resonator_feedline_circuit = ResonatorFeedline('quarter_wave')
    resonator_feedline_circuit.initialize()
    R1 = resonator_feedline_circuit.resonator_dict['R1']
    F1 = resonator_feedline_circuit.feedline_dict['F1']
    
    