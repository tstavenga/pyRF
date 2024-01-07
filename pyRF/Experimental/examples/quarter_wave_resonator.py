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

    eig = R1.get_eigenfunction(n=1)
    z = np.linspace(0,4e-3,200)
    plt.plot(z,eig(z))
    print(eig(0))
    ks = np.linspace(0,100,201,dtype=np.complex128)
    mc = np.array(list(map(lambda k:R1.mode_condition(k), ks)))
    print(mc)
    jac = np.array(list(map(lambda k:R1.matrix_condition_derivative(k), ks)))
    # print(R1.matrix_condition_derivative(100))
    fig, ax = plt.subplots(1,1)
    ax.plot(ks,mc[:,0], label='mode condition', color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(ks,np.real(jac), 'x',label = 'Jacobian', color='tab:orange')
    ax2.plot(ks,np.gradient(mc[:,0],ks[1]), label = 'Jacobian', color='tab:orange')

    ax.legend()
    ax2.legend()
    plt.show()
    # fig, ax = plt.subplots(1,1)
    # ax.plot(ks,mc[:,1], label='mode condition', color='tab:blue')
    # ax2 = ax.twinx()
    # ax2.plot(ks,np.imag(jac), label = 'Jacobian', color='tab:orange')
    # ax2.plot(ks,np.gradient(mc[:,1],ks[1]), label = 'Jacobian', color='tab:orange')
    # ax.legend()
    # ax2.legend()
    # # for n in range(3):
    # #     guess = 0
    # #     for i in range(15):
    # #         guess = R1.eigenvalue_guess(n+1,guess)
    # #     plt.plot(R1.get_eigenvalue(n=n+1),0,'rx')
    # #     plt.plot(guess,0,'g+')
    # plt.show()