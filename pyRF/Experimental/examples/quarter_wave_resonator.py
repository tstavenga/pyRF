from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt



class QuarterWave(Circuit):
    def __init__(self, name='Quarter wave'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 30e-10
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
    eig = R1.get_eigenfunction(n=1)
    z = np.linspace(0,4e-3,200)
    plt.plot(z,eig(z))
    print(eig(0))
    ks = np.linspace(0,100,201,dtype=np.complex128)
    mc = np.array(list(map(lambda k:R1.mode_condition(k), ks)))
    plt.figure()
    plt.plot(ks,mc)
    for n in range(3):
        guess = 0
        for i in range(15):
            guess = R1.eigenvalue_guess(n+1,guess)
        plt.plot(R1.get_eigenvalue(n=n+1),0,'rx')
        plt.plot(guess,0,'g+')
    plt.show()