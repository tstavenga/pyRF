from pyRF.circuit import Circuit
from pyRF.resonator import Resonator
from pyRF import node_element as ne
import numpy as np
import matplotlib.pyplot as plt



class TwoChannelResonator(Circuit):
    def __init__(self, name='two_channel_resonator'):
        super().__init__(name)

    def define_circuit_elements(self):
        capacitance = 29.68e-14
        open_capacitance = 24.5e-15
        open_position = 3547.4e-6
        capacitor_position = 3200e-6#3200e-6
        short_position = 0

        self.circuit_elements = {
            'O1': {
                'element':'Open',
                'values':{
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
                        'characteristic_impedance':50,
                        'phase_velocity':1e8,
                    },
                },
                'Capacitor_Short': {
                    'start_pin': {
                        'element': 'C1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'O1',
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
    two_channel_resonator = TwoChannelResonator('quarter_wave')
    two_channel_resonator.initialize()
    R1 = two_channel_resonator.resonator_dict['R1']
    ks = np.linspace(0,2500,20001,dtype=np.complex128)
    mc = np.array(list(map(R1.mode_condition, ks)))
    fig, ax = plt.subplots()
    # fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    print(R1.get_eigenvalue(n=9))
    ax.plot(ks,mc)
    ax3.plot(mc[:,0],mc[:,1])
    for n in range(25):
        el = R1.get_eigenvalue(n=n+1)
        guess = 0
        for i in range(150):
            guess = R1.eigenvalue_guess(n+1,guess)
        ax.plot(guess,0,'g+')
        # ax.plot(el,0,'rx')
        # ax2.plot(n,guess-el,'x')
        mcg = R1.mode_condition(guess)
        mce = R1.mode_condition(el)
        ax3.plot(mcg[0],mcg[1],'x')
        # ax3.plot(mce[0],mce[1],'x')
    ax.plot([min(ks),max(ks)],[0,0])
    plt.show()
